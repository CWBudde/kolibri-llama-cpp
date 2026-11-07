package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The command against a small fake of llama-server: flags, the API and the trace file.
func TestServes(t *testing.T) {
	turn := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		turn++
		if turn == 1 {
			fmt.Fprint(w, `data: {"choices":[{"finish_reason":null,"index":0,"delta":{"tool_calls":[{"index":0,"id":"x","type":"function","function":{"name":"int_add","arguments":"{\"a\":\"40\",\"b\":\"2\"}"}}]}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"finish_reason":"tool_calls","index":0,"delta":{}}]}`+"\n\ndata: [DONE]\n\n")
			return
		}
		fmt.Fprint(w, `data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"42"}}]}`+"\n\n")
		fmt.Fprint(w, `data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}]}`+"\n\ndata: [DONE]\n\n")
	}))
	defer upstream.Close()

	trace := filepath.Join(t.TempDir(), "trace.jsonl")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	addrs := make(chan string, 1)
	finished := make(chan error, 1)
	go func() {
		finished <- run(ctx, []string{"-listen", "127.0.0.1:0", "-upstream", upstream.URL, "-trace", trace, "-allow", "int_add,int_mul", "-max-turns", "3"},
			io.Discard, io.Discard, func(addr string) { addrs <- addr })
	}()
	var addr string
	select {
	case addr = <-addrs:
	case err := <-finished:
		t.Fatalf("run: %v", err)
	case <-time.After(10 * time.Second):
		t.Fatal("no listener")
	}

	resp, err := http.Post("http://"+addr+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"messages":[{"role":"user","content":"40 + 2?"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Bridge struct {
			Stop string `json:"stop"`
		} `json:"kolibri_bridge"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK || out.Choices[0].Message.Content != "42" || out.Bridge.Stop != "completed" {
		t.Fatalf("%d %+v", resp.StatusCode, out)
	}

	tools, err := http.Get("http://" + addr + "/v1/bridge/tools")
	if err != nil {
		t.Fatal(err)
	}
	defer tools.Body.Close()
	body, _ := io.ReadAll(tools.Body)
	if !strings.Contains(string(body), `"int_mul"`) || strings.Contains(string(body), `"hash_sha256"`) || !strings.Contains(string(body), `"max_turns":3`) {
		t.Errorf("manifest %s", body)
	}

	cancel()
	select {
	case err := <-finished:
		if err != nil {
			t.Errorf("run: %v", err)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("no shutdown")
	}

	raw, err := os.ReadFile(trace)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	for _, line := range lines {
		if !json.Valid([]byte(line)) {
			t.Errorf("trace line %q", line)
		}
	}
	if !strings.Contains(lines[len(lines)-1], `"kind":"request_end"`) || !strings.Contains(string(raw), `"op":"int_add"`) {
		t.Errorf("trace %s", raw)
	}
}

func TestListsOperations(t *testing.T) {
	var out bytes.Buffer
	if err := run(context.Background(), []string{"-ops"}, &out, io.Discard, nil); err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		Protocol int              `json:"protocol"`
		Ops      []map[string]any `json:"ops"`
	}
	if err := json.Unmarshal(out.Bytes(), &manifest); err != nil {
		t.Fatalf("%v: %s", err, out.String())
	}
	if manifest.Protocol != 1 || len(manifest.Ops) != 15 {
		t.Errorf("protocol %d, %d operations", manifest.Protocol, len(manifest.Ops))
	}
}

func TestBadFlags(t *testing.T) {
	for _, args := range [][]string{
		{"-allow", "typo"},
		{"-max-turns", "0"},
		{"-max-int-bits", "0"},
		{"-upstream", ""},
		{"stray"},
		{"-listen", "not an address"},
	} {
		if err := run(context.Background(), args, io.Discard, io.Discard, nil); err == nil {
			t.Errorf("%q passes", args)
		}
	}
}
