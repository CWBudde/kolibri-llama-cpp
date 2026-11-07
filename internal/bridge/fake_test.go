package bridge

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"kolibri-llm/internal/bridge/ops"
)

// fakeCall is one tool call in a scripted turn.
type fakeCall struct{ name, args string }

// fakeTurn scripts one completion.
type fakeTurn struct {
	reasoning string
	content   string
	calls     []fakeCall
	finish    string // default: tool_calls with calls, stop without

	raw     string // replays these bytes as the response, a captured stream for example
	rawCT   string // their content type; default text/event-stream
	status  int    // the status of raw; default 200
	rawHang bool   // blocks after raw until the client goes away

	cutAfter  int    // ends the connection after this many events, 0 means don't
	cutLine   bool   // ... in the middle of the next one
	noDone    bool   // leaves out [DONE]
	hang      bool   // blocks until the client goes away
	hangAfter int    // ... after this many events
	onRequest func() // runs when the request arrives
}

// fakeLlama is llama-server's /v1/chat/completions as far as the bridge can tell. It streams in the
// shapes of testdata/bridge, which TestFakeHasTheCapturedShapes checks: a role chunk first, argument
// fragments of a few characters, a finish chunk with an empty delta, a usage chunk without choices,
// [DONE]. Its call IDs are random like llama-server's.
type fakeLlama struct {
	t      *testing.T
	srv    *httptest.Server
	script func(turn int, body []byte) fakeTurn

	mu       sync.Mutex
	requests [][]byte
	gone     chan struct{} // closed when a hanging request's client has left
	goneOnce sync.Once
}

func newFake(t *testing.T, turns ...fakeTurn) *fakeLlama {
	t.Helper()
	return newFakeFunc(t, func(turn int, _ []byte) fakeTurn {
		if turn >= len(turns) {
			t.Errorf("turn %d, the script has %d", turn+1, len(turns))
			return fakeTurn{content: "script ended"}
		}
		return turns[turn]
	})
}

func newFakeFunc(t *testing.T, script func(turn int, body []byte) fakeTurn) *fakeLlama {
	t.Helper()
	f := &fakeLlama{t: t, script: script, gone: make(chan struct{})}
	f.srv = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeLlama) requestCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.requests)
}

func (f *fakeLlama) request(i int) []byte {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.requests[i]
}

// messages decodes the messages of the i-th request.
func (f *fakeLlama) messages(i int) []map[string]any {
	f.t.Helper()
	var body struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(f.request(i), &body); err != nil {
		f.t.Fatal(err)
	}
	return body.Messages
}

func randomID() string {
	const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, 32)
	for i := range b {
		b[i] = alphabet[rand.IntN(len(alphabet))]
	}
	return string(b)
}

// fragments cuts text the way the captured streams do: a few characters at a time, never inside a
// UTF-8 sequence, which llama-server holds back until it is complete.
func fragments(s string) []string {
	var out []string
	for len(s) > 0 {
		n := min(3, len(s))
		for n < len(s) && !utf8.RuneStart(s[n]) {
			n++
		}
		out = append(out, s[:n])
		s = s[n:]
	}
	return out
}

func (f *fakeLlama) events(turn fakeTurn) []string {
	id := "chatcmpl-" + randomID()
	event := func(extra string) string {
		return fmt.Sprintf(`{%s,"created":1794010650,"id":%q,"model":"kolibri-tiny-f32.gguf","system_fingerprint":"b11839-90311e5a2","object":"chat.completion.chunk"%s}`,
			"%s", id, extra)
	}
	delta := func(d string) string {
		return fmt.Sprintf(event(""), fmt.Sprintf(`"choices":[{"finish_reason":null,"index":0,"delta":%s}]`, d))
	}
	str := func(s string) string { b, _ := json.Marshal(s); return string(b) }

	out := []string{delta(`{"role":"assistant","content":null}`)}
	for _, part := range fragments(turn.reasoning) {
		out = append(out, delta(`{"reasoning_content":`+str(part)+`}`))
	}
	for _, part := range fragments(turn.content) {
		out = append(out, delta(`{"content":`+str(part)+`}`))
	}
	for i, c := range turn.calls {
		parts := fragments(c.args)
		first := ""
		if len(parts) > 0 {
			first, parts = parts[0], parts[1:]
		}
		out = append(out, delta(fmt.Sprintf(`{"tool_calls":[{"index":%d,"id":%q,"type":"function","function":{"name":%s,"arguments":%s}}]}`,
			i, randomID(), str(c.name), str(first))))
		for _, part := range parts {
			out = append(out, delta(fmt.Sprintf(`{"tool_calls":[{"index":%d,"function":{"arguments":%s}}]}`, i, str(part))))
		}
	}
	finish := turn.finish
	if finish == "" {
		finish = "stop"
		if len(turn.calls) > 0 {
			finish = "tool_calls"
		}
	}
	out = append(out, fmt.Sprintf(event(""), fmt.Sprintf(`"choices":[{"finish_reason":%q,"index":0,"delta":{}}]`, finish)))
	out = append(out, fmt.Sprintf(event(`,"usage":{"completion_tokens":5,"prompt_tokens":10,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":7}},"timings":{"predicted_n":5}`), `"choices":[]`))
	return out
}

func (f *fakeLlama) serve(w http.ResponseWriter, r *http.Request) {
	body := make([]byte, 0, 4096)
	buf := make([]byte, 4096)
	for {
		n, err := r.Body.Read(buf)
		body = append(body, buf[:n]...)
		if err != nil {
			break
		}
	}
	f.mu.Lock()
	turn := len(f.requests)
	f.requests = append(f.requests, body)
	f.mu.Unlock()

	if r.URL.Path != "/v1/chat/completions" {
		f.t.Errorf("request to %s", r.URL.Path)
	}
	script := f.script(turn, body)
	if script.onRequest != nil {
		script.onRequest()
	}
	waitForClient := func() {
		<-r.Context().Done()
		f.goneOnce.Do(func() { close(f.gone) })
	}
	if script.hang {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		waitForClient()
		return
	}
	if script.raw != "" {
		ct := script.rawCT
		if ct == "" {
			ct = "text/event-stream"
		}
		w.Header().Set("Content-Type", ct)
		if script.status != 0 {
			w.WriteHeader(script.status)
		}
		_, _ = w.Write([]byte(script.raw))
		if script.rawHang {
			w.(http.Flusher).Flush()
			waitForClient()
		}
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	flusher := w.(http.Flusher)
	for i, event := range f.events(script) {
		if script.cutAfter > 0 && i == script.cutAfter {
			if script.cutLine {
				_, _ = w.Write([]byte("data: " + event[:len(event)/2]))
				flusher.Flush()
			}
			panic(http.ErrAbortHandler)
		}
		if script.hangAfter > 0 && i == script.hangAfter {
			flusher.Flush()
			waitForClient()
			return
		}
		_, _ = w.Write([]byte("data: " + event + "\n\n"))
		flusher.Flush()
	}
	if !script.noDone {
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}
}

func fixture(t *testing.T, name string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "bridge", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

// testLimits are small, so that a limit is reached quickly and its failure is obvious.
func testLimits() Limits {
	lim := DefaultLimits()
	lim.MaxTurns = 6
	lim.MaxCalls = 12
	lim.MaxCallsPerTurn = 4
	lim.OpTimeout = 2 * time.Second
	lim.TurnTimeout = 5 * time.Second
	lim.RequestTimeout = 30 * time.Second
	return lim
}

// newBridge starts a bridge on the fake, with changes to its config.
func newBridge(t *testing.T, f *fakeLlama, change ...func(*Config)) *Bridge {
	t.Helper()
	cfg := Config{Upstream: f.srv.URL, Limits: testLimits(), OpLimits: ops.DefaultLimits()}
	for _, c := range change {
		c(&cfg)
	}
	b, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := b.Close(ctx); err != nil {
			t.Errorf("close: %v", err)
		}
	})
	return b
}

const userBody = `{"messages":[{"role":"user","content":"Go on."}],"temperature":0}`

func runBody(t *testing.T, b *Bridge, body string) *Result {
	t.Helper()
	res, err := b.Run(context.Background(), "r", []byte(body), Callbacks{})
	if err != nil {
		t.Fatal(err)
	}
	return res
}

// checkTrace holds for every request, however it ended: numbered events, one start, one end that
// says the same as the result, turns that pair up, and a result for every call that started before
// the request stopped running.
func checkTrace(t *testing.T, res *Result) {
	t.Helper()
	if len(res.Trace) < 2 {
		t.Fatalf("trace %+v", res.Trace)
	}
	if res.Trace[0].Kind != EventRequestStart || res.Trace[0].Protocol != ops.Version {
		t.Errorf("first event %+v", res.Trace[0])
	}
	last := res.Trace[len(res.Trace)-1]
	if last.Kind != EventRequestEnd || last.Stop != res.Stop {
		t.Errorf("last event %+v, result stop %s", last, res.Stop)
	}
	turns, inTurn := 0, 0
	calls := map[string]bool{} // started calls, and whether each has its result
	for i, e := range res.Trace {
		if e.Seq != i+1 || e.RequestID != res.RequestID {
			t.Errorf("event %d: %+v", i, e)
		}
		switch e.Kind {
		case EventTurnStart:
			if inTurn != 0 {
				t.Errorf("event %d: turn %d starts inside turn %d", i, e.Turn, inTurn)
			}
			turns++
			inTurn = e.Turn
		case EventTurnEnd:
			if e.Turn != inTurn {
				t.Errorf("event %d: turn %d ends inside turn %d", i, e.Turn, inTurn)
			}
			inTurn = 0
		case EventCallStart:
			if _, seen := calls[e.CallID]; seen {
				t.Errorf("event %d: call %s starts twice", i, e.CallID)
			}
			calls[e.CallID] = false
		case EventCallEnd:
			if done, seen := calls[e.CallID]; !seen || done {
				t.Errorf("event %d: call %s ends without a running start", i, e.CallID)
			}
			calls[e.CallID] = true
		}
	}
	if inTurn != 0 {
		t.Errorf("turn %d never ended", inTurn)
	}
	if turns != res.Turns {
		t.Errorf("%d turn_start events, result says %d turns", turns, res.Turns)
	}
	// Only a request stopped from outside can leave a call without a result.
	if res.Stop != StopCancelled && res.Stop != StopRequestTimeout {
		for id, done := range calls {
			if !done {
				t.Errorf("call %s has no result, stop %s", id, res.Stop)
			}
		}
	}
}

// callEnds lists the results the model got to see, in order.
func callEnds(res *Result) []Event {
	var out []Event
	for _, e := range res.Trace {
		if e.Kind == EventCallEnd {
			out = append(out, e)
		}
	}
	return out
}

func wantStop(t *testing.T, res *Result, want StopReason) {
	t.Helper()
	checkTrace(t, res)
	if res.Stop != want {
		t.Fatalf("stop %s, want %s: %+v", res.Stop, want, res.Failure)
	}
	if (res.Failure == nil) != (want == StopCompleted) {
		t.Fatalf("stop %s with failure %+v", res.Stop, res.Failure)
	}
}

// keyPaths lists a JSON value's key paths: what "the same shape" means here.
func keyPaths(prefix string, v any, into map[string]bool) {
	switch v := v.(type) {
	case map[string]any:
		for k, x := range v {
			keyPaths(prefix+"."+k, x, into)
		}
	case []any:
		for _, x := range v {
			keyPaths(prefix+"[]", x, into)
		}
	default:
		into[prefix] = true
	}
}

func streamShapes(t *testing.T, stream string) map[string]bool {
	t.Helper()
	shapes := map[string]bool{}
	for line := range strings.Lines(stream) {
		data, ok := strings.CutPrefix(strings.TrimSpace(line), "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var v any
		if err := json.Unmarshal([]byte(data), &v); err != nil {
			t.Fatalf("%v: %s", err, data)
		}
		keyPaths("", v, shapes)
	}
	return shapes
}

// The fake's stream has the key paths of the real streams, and only those. timings holds llama-
// server's speed figures, which the bridge doesn't read and the fake needn't list.
func TestFakeHasTheCapturedShapes(t *testing.T) {
	real := map[string]bool{}
	for _, name := range []string{"tool_call.sse", "answer_len.sse", "reasoning.sse"} {
		for p := range streamShapes(t, fixture(t, name)) {
			real[p] = true
		}
	}
	f := newFake(t)
	var stream strings.Builder
	for _, e := range f.events(fakeTurn{reasoning: "thinking", content: "text", calls: []fakeCall{{"int_add", `{"a":"1","b":"2"}`}}}) {
		stream.WriteString("data: " + e + "\n\n")
	}
	fake := streamShapes(t, stream.String())
	strip := func(m map[string]bool) []string {
		var out []string
		for p := range m {
			if !strings.HasPrefix(p, ".timings") {
				out = append(out, p)
			}
		}
		slices.Sort(out)
		return out
	}
	if got, want := strip(fake), strip(real); !slices.Equal(got, want) {
		t.Errorf("the fake's shape differs from the captured one:\nfake %q\nreal %q", got, want)
	}
}
