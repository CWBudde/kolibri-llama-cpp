package bridge

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"kolibri-llm/internal/bridge/ops"
	"kolibri-llm/internal/coproc"
)

// running is the scheduler's state of a running request, Closed if it isn't running.
func (b *Bridge) running(id string) coproc.RequestState {
	b.mu.Lock()
	defer b.mu.Unlock()
	if r := b.runs[id]; r != nil && r.req != nil {
		return r.req.State()
	}
	return coproc.Closed
}

// jobState is the scheduler's state of a call of a running request.
func (b *Bridge) jobState(id, call string) coproc.JobState {
	b.mu.Lock()
	defer b.mu.Unlock()
	if r := b.runs[id]; r != nil && r.req != nil {
		if info, ok := r.req.Job(coproc.JobID(call)); ok {
			return info.State
		}
	}
	return coproc.JobIdle
}

// A multi-step task: each step's input is the previous step's output, as the model read it. The
// fake is scripted to look at the tool message it got and to use that value, so a result that didn't
// reach the next prompt fails the chain.
func TestMultiStepChain(t *testing.T) {
	wantHash := sha256.Sum256([]byte("121932631356500531347203169112635270"))
	hash := hex.EncodeToString(wantHash[:])
	lastResult := func(body []byte) string {
		var req struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			t.Error(err)
		}
		var resp struct {
			Result struct {
				Value string `json:"value"`
				Hex   string `json:"hex"`
			} `json:"result"`
		}
		if err := json.Unmarshal([]byte(req.Messages[len(req.Messages)-1].Content), &resp); err != nil {
			t.Errorf("last message isn't a response: %v", err)
		}
		return resp.Result.Value + resp.Result.Hex
	}
	f := newFakeFunc(t, func(turn int, body []byte) fakeTurn {
		switch turn {
		case 0:
			return fakeTurn{reasoning: "Multiply first.", calls: []fakeCall{{"int_mul", `{"a":"987654321987654321","b":"123456789123456789"}`}}}
		case 1:
			return fakeTurn{reasoning: "Add one.", calls: []fakeCall{{"int_add", fmt.Sprintf(`{"a":%q,"b":"1"}`, lastResult(body))}}}
		case 2:
			return fakeTurn{content: "Now the hash.", calls: []fakeCall{{"hash_sha256", fmt.Sprintf(`{"data":%q}`, lastResult(body))}}}
		}
		return fakeTurn{reasoning: "Done.", content: "The hash is " + lastResult(body)}
	})
	b := newBridge(t, f)
	res := runBody(t, b, userBody)
	wantStop(t, res, StopCompleted)
	if res.Content != "The hash is "+hash || res.Reasoning != "Done." || res.Finish != "stop" {
		t.Errorf("answer %q, reasoning %q, finish %q", res.Content, res.Reasoning, res.Finish)
	}
	if res.Turns != 4 || res.Calls != 3 || f.requestCount() != 4 {
		t.Errorf("%d turns, %d calls, %d requests", res.Turns, res.Calls, f.requestCount())
	}
	if res.Usage.TotalTokens != 60 || res.Usage.CachedTokens != 28 {
		t.Errorf("usage %+v", res.Usage)
	}

	// The prompt of the last turn, message by message, in the shape check_chat.py renders.
	got := f.messages(3)
	if len(got) != 7 {
		t.Fatalf("%d messages: %v", len(got), got)
	}
	wantRoles := []string{"user", "assistant", "tool", "assistant", "tool", "assistant", "tool"}
	for i, role := range wantRoles {
		if got[i]["role"] != role {
			t.Errorf("message %d: role %v, want %s", i, got[i]["role"], role)
		}
	}
	// Reasoning goes back: the template renders it between the last query and the tool call.
	if got[1]["reasoning_content"] != "Multiply first." || got[5]["content"] != "Now the hash." {
		t.Errorf("assistant messages %v, %v", got[1], got[5])
	}
	if got[1]["tool_calls"].([]any)[0].(map[string]any)["id"] != "t1c0" || got[2]["tool_call_id"] != "t1c0" {
		t.Errorf("call IDs %v, %v", got[1], got[2])
	}
	if want := `{"v":1,"id":"t1c0","op":"int_mul","ok":true,"result":{"value":"121932631356500531347203169112635269"}}`; got[2]["content"] != want {
		t.Errorf("first result %v", got[2]["content"])
	}

	// The trace names every step in order.
	var kinds []string
	for _, e := range res.Trace {
		kinds = append(kinds, string(e.Kind))
	}
	want := []string{"request_start"}
	for range 3 {
		want = append(want, "turn_start", "turn_end", "call_start", "call_end")
	}
	want = append(want, "turn_start", "turn_end", "request_end")
	if !reflect.DeepEqual(kinds, want) {
		t.Errorf("trace kinds %v", kinds)
	}
	ends := callEnds(res)
	if ends[2].Op != "hash_sha256" || ends[2].JobState != "DONE" || ends[2].Code != "" || ends[2].ResultSHA256 == "" {
		t.Errorf("call_end %+v", ends[2])
	}
}

// Each request carries the tools, a stream with usage, and what the client set, and nothing changes
// between turns but the messages.
func TestRequestShape(t *testing.T) {
	f := newFake(t, fakeTurn{calls: []fakeCall{{"int_add", `{"a":"1","b":"2"}`}}}, fakeTurn{content: "3"})
	b := newBridge(t, f, func(c *Config) { c.Allow = []string{"int_add", "hash_sha256"} })
	res := runBody(t, b, `{"messages":[{"role":"user","content":"hi"}],"temperature":0,"seed":7,"reasoning_effort":"low","max_tokens":99}`)
	wantStop(t, res, StopCompleted)
	var first, second map[string]json.RawMessage
	if err := json.Unmarshal(f.request(0), &first); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(f.request(1), &second); err != nil {
		t.Fatal(err)
	}
	for field, want := range map[string]string{
		"temperature": "0", "seed": "7", "reasoning_effort": `"low"`, "max_tokens": "99", "stream": "true",
		"stream_options": `{"include_usage":true}`,
	} {
		if string(first[field]) != want {
			t.Errorf("%s = %s, want %s", field, first[field], want)
		}
	}
	var tools []struct {
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(first["tools"], &tools); err != nil {
		t.Fatal(err)
	}
	if len(tools) != 2 || tools[0].Function.Name != "hash_sha256" || tools[1].Function.Name != "int_add" {
		t.Errorf("tools %s", first["tools"])
	}
	delete(first, "messages")
	delete(second, "messages")
	if !reflect.DeepEqual(first, second) {
		t.Errorf("turns differ:\n%s\n%s", first, second)
	}
}

// The real server's bytes, captured from llama-server on the tiny checkpoint: a forced call, with
// the whitespace the grammar lets a random model write, then an answer.
func TestReplayOfACapturedToolLoop(t *testing.T) {
	f := newFake(t, fakeTurn{raw: fixture(t, "tool_call.sse")}, fakeTurn{raw: fixture(t, "answer.sse")})
	b := newBridge(t, f)
	res := runBody(t, b, userBody)
	wantStop(t, res, StopCompleted)
	ends := callEnds(res)
	if len(ends) != 1 || ends[0].Result != `{"v":1,"id":"t1c0","op":"int_add","ok":true,"result":{"value":"144431509"}}` {
		t.Fatalf("results %+v", ends)
	}
	if res.Usage.CompletionTokens != 61 {
		t.Errorf("usage %+v", res.Usage)
	}
	msgs := f.messages(1)
	fn := msgs[1]["tool_calls"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if want := "{\"a\"\n                    :\"78\", \"b\":\"144431431\"}"; fn["arguments"] != want {
		t.Errorf("arguments changed on the way back: %q", fn["arguments"])
	}
}

// The non-streaming shape of the same call, for a server that doesn't stream.
func TestReplayOfANonStreamingResponse(t *testing.T) {
	f := newFake(t,
		fakeTurn{raw: fixture(t, "tool_call.json"), rawCT: "application/json; charset=utf-8"},
		fakeTurn{raw: fixture(t, "second.json"), rawCT: "application/json; charset=utf-8"})
	b := newBridge(t, f)
	res := runBody(t, b, userBody)
	wantStop(t, res, StopCompleted)
	if ends := callEnds(res); len(ends) != 1 || !strings.Contains(ends[0].Result, `"value":"144431509"`) {
		t.Fatalf("results %+v", ends)
	}
}

// Captured: a turn out of tokens with one complete call and a second that has only its "{". Nothing
// runs, not even the complete one.
func TestCapturedCutTurnExecutesNothing(t *testing.T) {
	for _, name := range []string{"tool_call_cut.sse", "answer_len.sse"} {
		t.Run(name, func(t *testing.T) {
			f := newFake(t, fakeTurn{raw: fixture(t, name)})
			res := runBody(t, newBridge(t, f), userBody)
			wantStop(t, res, StopTruncated)
			if res.Calls != 0 || len(callEnds(res)) != 0 || f.requestCount() != 1 {
				t.Fatalf("%d calls, %d requests", res.Calls, f.requestCount())
			}
			// What the model asked for stays visible in the trace, marked as refused.
			var refused []Event
			for _, e := range res.Trace {
				switch e.Kind {
				case EventCallRefused:
					refused = append(refused, e)
				case EventCallStart:
					t.Errorf("a call started: %+v", e)
				}
			}
			if name == "tool_call_cut.sse" && (len(refused) != 2 || refused[0].Op != "int_add" || refused[1].Args != "{" || refused[0].Code != "truncated") {
				t.Errorf("refused %+v", refused)
			}
		})
	}
}

// Captured: output the parser can't read is an error event in the stream, then the connection ends.
func TestCapturedParseErrorMidStream(t *testing.T) {
	f := newFake(t, fakeTurn{raw: fixture(t, "parse_error.sse")})
	res := runBody(t, newBridge(t, f), userBody)
	wantStop(t, res, StopUpstreamError)
	if res.Failure.UpstreamType != "server_error" || res.Failure.HTTPStatus != 500 ||
		!strings.Contains(res.Failure.Message, "peg-native") {
		t.Fatalf("failure %+v", res.Failure)
	}
}

// Captured: the server's refusals carry its error type.
func TestCapturedRefusals(t *testing.T) {
	for _, c := range []struct {
		name, kind string
		status     int
	}{
		{"exceed_context.json", "exceed_context_size_error", 400},
		{"bad_request.json", "invalid_request_error", 400},
		{"no_jinja.json", "server_error", 500},
		{"parse_error.json", "server_error", 500},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := newFake(t, fakeTurn{raw: fixture(t, c.name), rawCT: "application/json; charset=utf-8", status: c.status})
			res := runBody(t, newBridge(t, f), userBody)
			wantStop(t, res, StopUpstreamHTTP)
			if res.Failure.UpstreamType != c.kind || res.Failure.HTTPStatus != c.status || res.Failure.Message == "" {
				t.Fatalf("failure %+v", res.Failure)
			}
		})
	}
}

// A call that fails goes back to the model, machine-readable, and the model goes on to answer. The
// table is what a model can get wrong.
func TestBadCallsGoBackToTheModel(t *testing.T) {
	for _, c := range []struct {
		name, op, args string
		code           ops.Code
	}{
		{"not json", "int_add", `nonsense`, ops.CodeMalformedArguments},
		{"cut json", "int_add", `{"a":"1"`, ops.CodeMalformedArguments},
		{"empty", "int_add", ``, ops.CodeMalformedArguments},
		{"array", "int_add", `[1,2]`, ops.CodeMalformedArguments},
		{"duplicate key", "int_add", `{"a":"1","a":"2","b":"3"}`, ops.CodeMalformedArguments},
		{"missing field", "int_add", `{"a":"1"}`, ops.CodeInvalidArgument},
		{"extra field", "int_add", `{"a":"1","b":"2","c":"3"}`, ops.CodeInvalidArgument},
		{"float", "int_add", `{"a":1.5,"b":"2"}`, ops.CodeInvalidArgument},
		{"rounded by the server", "int_add", `{"a":1.8446744073709552e+19,"b":"2"}`, ops.CodeInvalidArgument},
		{"past int64", "int_add", `{"a":9223372036854775808,"b":"2"}`, ops.CodeInvalidArgument},
		{"unknown op", "read_file", `{"path":"/etc/passwd"}`, ops.CodeUnknownOp},
		{"no name", "", `{}`, ops.CodeUnknownOp},
		{"not allowed", "int_mul", `{"a":"2","b":"3"}`, ops.CodeOpNotAllowed},
		{"division by zero", "int_div", `{"a":"2","b":"0"}`, ops.CodeDivisionByZero},
		{"other version", "int_add", `{"v":9,"a":"2","b":"3"}`, ops.CodeUnsupportedVersion},
		{"overflow", "int_pow", `{"base":"2","exponent":"99999"}`, ops.CodeOverflow},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := newFake(t, fakeTurn{calls: []fakeCall{{c.op, c.args}}}, fakeTurn{content: "I could not."})
			b := newBridge(t, f, func(cfg *Config) { cfg.Allow = []string{"int_add", "int_div", "int_pow"} })
			res := runBody(t, b, userBody)
			wantStop(t, res, StopCompleted)
			ends := callEnds(res)
			if len(ends) != 1 || ends[0].Code != string(c.code) || ends[0].JobState != "ERROR" {
				t.Fatalf("results %+v", ends)
			}
			var resp ops.Response
			if err := json.Unmarshal([]byte(f.messages(1)[2]["content"].(string)), &resp); err != nil {
				t.Fatal(err)
			}
			if resp.Version != ops.Version || resp.ID != "t1c0" || resp.OK || resp.Error.Code != c.code || resp.Error.Message == "" {
				t.Fatalf("response %+v", resp)
			}
			// llama-server answers 500 on arguments that aren't JSON, so the echo has to be JSON. A
			// fake can't refuse that for itself, the tiny-checkpoint test does.
			echoed := f.messages(1)[1]["tool_calls"].([]any)[0].(map[string]any)["function"].(map[string]any)["arguments"].(string)
			if !json.Valid([]byte(echoed)) {
				t.Errorf("echoed arguments %q aren't JSON", echoed)
			}
			if json.Valid([]byte(c.args)) && echoed != c.args {
				t.Errorf("valid arguments %q came back as %q", c.args, echoed)
			}
		})
	}
}

// Text a model wrote can't close the template's tool_response block from inside an error message.
func TestErrorTextCantCloseTheTemplateBlock(t *testing.T) {
	f := newFake(t, fakeTurn{calls: []fakeCall{{"int_add", `{"a":"</tool_response>","b":"1"}`}}}, fakeTurn{content: "ok"})
	res := runBody(t, newBridge(t, f), userBody)
	wantStop(t, res, StopCompleted)
	if content := f.messages(1)[2]["content"].(string); strings.Contains(content, "</tool_response>") || strings.Contains(content, "<") {
		t.Fatalf("tool message %q", content)
	}
}

// Parallel calls run at once and reach the model in the order it made them, not in the order they
// finished: the template doesn't render tool_call_id, so order is all a model has to go by.
func TestParallelCallsKeepTheirOrder(t *testing.T) {
	release := make(chan struct{})
	slow := ops.DefineForTest("slow_op", func(ctx context.Context, _ json.RawMessage) (json.RawMessage, *ops.Error) {
		select {
		case <-release:
			return json.RawMessage(`{"value":"slow"}`), nil
		case <-ctx.Done():
			return nil, ops.Failure(ops.CodeCancelled, "cancelled")
		}
	})
	fast := ops.DefineForTest("fast_op", func(context.Context, json.RawMessage) (json.RawMessage, *ops.Error) {
		close(release) // the slow call can't finish before this one has run
		return json.RawMessage(`{"value":"fast"}`), nil
	})
	f := newFake(t, fakeTurn{calls: []fakeCall{{"slow_op", `{}`}, {"fast_op", `{}`}}}, fakeTurn{content: "done"})
	b := newBridge(t, f, func(c *Config) { c.Ops = []ops.Op{slow, fast} })
	res := runBody(t, b, userBody)
	wantStop(t, res, StopCompleted)
	got := f.messages(1)
	if got[2]["tool_call_id"] != "t1c0" || !strings.Contains(got[2]["content"].(string), "slow") ||
		got[3]["tool_call_id"] != "t1c1" || !strings.Contains(got[3]["content"].(string), "fast") {
		t.Fatalf("results out of order: %v", got[2:])
	}
}

func slowOp(started chan<- struct{}, gotCancel *atomic.Int32) ops.Op {
	return ops.DefineForTest("slow_op", func(ctx context.Context, _ json.RawMessage) (json.RawMessage, *ops.Error) {
		if started != nil {
			started <- struct{}{}
		}
		<-ctx.Done()
		if gotCancel != nil {
			gotCancel.Add(1)
		}
		return nil, ops.Failure(ops.CodeCancelled, "stopped")
	})
}

// An operation past its deadline is the error "timeout", the worker's context ends, and the model
// gets to answer. What the worker reports afterwards changes nothing.
func TestOperationTimeout(t *testing.T) {
	var stopped atomic.Int32
	f := newFake(t, fakeTurn{calls: []fakeCall{{"slow_op", `{}`}}}, fakeTurn{content: "too slow"})
	b := newBridge(t, f, func(c *Config) {
		c.Ops = []ops.Op{slowOp(nil, &stopped)}
		c.Limits.OpTimeout = 30 * time.Millisecond
	})
	res := runBody(t, b, userBody)
	wantStop(t, res, StopCompleted)
	ends := callEnds(res)
	if len(ends) != 1 || ends[0].Code != "timeout" || ends[0].JobState != "TIMEOUT" {
		t.Fatalf("results %+v", ends)
	}
	if content := f.messages(1)[2]["content"].(string); !strings.Contains(content, `"code":"timeout"`) {
		t.Fatalf("tool message %s", content)
	}
	waitFor(t, func() bool { return stopped.Load() == 1 })
}

// waitFor polls for a condition that has no event to wait on. It reports with t.Error, so that a
// goroutine other than the test's can use it too.
func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Error("condition not met in time")
			return
		}
		runtime.Gosched()
		time.Sleep(time.Millisecond)
	}
}

// Cancelling the whole request while an operation runs ends both, with the reason "cancelled".
func TestCancelDuringAnOperation(t *testing.T) {
	started := make(chan struct{}, 1)
	var stopped atomic.Int32
	f := newFake(t, fakeTurn{calls: []fakeCall{{"slow_op", `{}`}}})
	b := newBridge(t, f, func(c *Config) { c.Ops = []ops.Op{slowOp(started, &stopped)} })
	go func() {
		<-started
		if !b.Cancel("r") {
			t.Error("the request isn't running")
		}
	}()
	res := runBody(t, b, userBody)
	wantStop(t, res, StopCancelled)
	if f.requestCount() != 1 {
		t.Errorf("%d requests after a cancel", f.requestCount())
	}
	waitFor(t, func() bool { return stopped.Load() == 1 })
	var requested bool
	for _, e := range res.Trace {
		requested = requested || e.Kind == EventCancelRequested
	}
	if !requested {
		t.Error("the trace doesn't show the cancel")
	}
	if b.Cancel("r") {
		t.Error("a finished request can still be cancelled")
	}
}

// Cancelling one call is that call's result, and the request goes on.
func TestCancelOneCall(t *testing.T) {
	started := make(chan struct{}, 1)
	f := newFake(t, fakeTurn{calls: []fakeCall{{"slow_op", `{}`}}}, fakeTurn{content: "fine"})
	b := newBridge(t, f, func(c *Config) { c.Ops = []ops.Op{slowOp(started, nil)} })
	go func() {
		<-started
		if cancelled, err := b.CancelCall("r", "t1c0"); err != nil || !cancelled {
			t.Errorf("cancelled %v, %v", cancelled, err)
		}
	}()
	res := runBody(t, b, userBody)
	wantStop(t, res, StopCompleted)
	ends := callEnds(res)
	if len(ends) != 1 || ends[0].Code != "cancelled" || ends[0].JobState != "CANCELLED" {
		t.Fatalf("results %+v", ends)
	}
	if _, err := b.CancelCall("r", "t1c0"); err == nil {
		t.Error("a call of a finished request can still be cancelled")
	}
}

// A call that already has its result keeps it: the first of two parallel calls is done, the second
// still runs when the cancel arrives.
func TestCancelACallThatIsDone(t *testing.T) {
	started := make(chan struct{}, 1)
	fast := ops.DefineForTest("fast_op", func(context.Context, json.RawMessage) (json.RawMessage, *ops.Error) {
		return json.RawMessage(`{"value":"fast"}`), nil
	})
	f := newFake(t, fakeTurn{calls: []fakeCall{{"fast_op", `{}`}, {"slow_op", `{}`}}}, fakeTurn{content: "fine"})
	b := newBridge(t, f, func(c *Config) { c.Ops = []ops.Op{fast, slowOp(started, nil)} })
	errs := make(chan error, 1)
	go func() {
		<-started
		// The fast call's completion may still be on its way to the scheduler.
		waitFor(t, func() bool { return b.jobState("r", "t1c0") == coproc.JobDone })
		got, err := b.CancelCall("r", "t1c0")
		if got || err != nil {
			errs <- fmt.Errorf("a done call: cancelled %v, %v", got, err)
			return
		}
		got, err = b.CancelCall("r", "t1c1")
		if !got || err != nil {
			errs <- fmt.Errorf("a running call: cancelled %v, %v", got, err)
			return
		}
		errs <- nil
	}()
	res := runBody(t, b, userBody)
	wantStop(t, res, StopCompleted)
	if err := <-errs; err != nil {
		t.Error(err)
	}
	ends := callEnds(res)
	if len(ends) != 2 || ends[0].Code != "" || ends[1].Code != "cancelled" {
		t.Fatalf("results %+v", ends)
	}
}

// While an operation runs, no completion does: the request is BLOCKED and nothing asks the server
// for tokens until the result is there.
func TestNoTurnWhileAnOperationRuns(t *testing.T) {
	started := make(chan struct{}, 1)
	release := make(chan struct{})
	op := ops.DefineForTest("slow_op", func(context.Context, json.RawMessage) (json.RawMessage, *ops.Error) {
		started <- struct{}{}
		<-release
		return json.RawMessage(`{"value":"done"}`), nil
	})
	f := newFake(t, fakeTurn{calls: []fakeCall{{"slow_op", `{}`}}}, fakeTurn{content: "done"})
	b := newBridge(t, f, func(c *Config) { c.Ops = []ops.Op{op} })
	results := make(chan *Result, 1)
	go func() { results <- runBody(t, b, userBody) }()
	<-started
	waitFor(t, func() bool { return b.running("r") == coproc.Blocked })
	if n := f.requestCount(); n != 1 {
		t.Errorf("%d requests while the operation runs", n)
	}
	close(release)
	wantStop(t, <-results, StopCompleted)
	if n := f.requestCount(); n != 2 {
		t.Errorf("%d requests", n)
	}
}

func TestCancelUnknownCallOfARunningRequest(t *testing.T) {
	started := make(chan struct{}, 1)
	f := newFake(t, fakeTurn{calls: []fakeCall{{"slow_op", `{}`}}}, fakeTurn{content: "fine"})
	b := newBridge(t, f, func(c *Config) {
		c.Ops = []ops.Op{slowOp(started, nil)}
		c.Limits.OpTimeout = 300 * time.Millisecond
	})
	errs := make(chan error, 1)
	go func() {
		<-started
		_, err := b.CancelCall("r", "t9c9")
		errs <- err
	}()
	res := runBody(t, b, userBody)
	wantStop(t, res, StopCompleted)
	if err := <-errs; err == nil {
		t.Error("an unknown call can be cancelled")
	}
}

// Cancelling during a turn cuts the upstream connection: llama-server stops generating when its
// client leaves. The reason is the cancel, not the broken stream it causes.
func TestCancelDuringATurn(t *testing.T) {
	for _, c := range []struct {
		name string
		turn fakeTurn
	}{
		{"before the first event", fakeTurn{hang: true}},
		{"inside a call", fakeTurn{calls: []fakeCall{{"int_add", `{"a":"1","b":"2"}`}}, hangAfter: 3}},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := newFake(t, c.turn)
			b := newBridge(t, f)
			go func() {
				waitFor(t, func() bool { return f.requestCount() == 1 })
				b.Cancel("r")
			}()
			res := runBody(t, b, userBody)
			wantStop(t, res, StopCancelled)
			if res.Calls != 0 {
				t.Errorf("%d calls ran", res.Calls)
			}
			select {
			case <-f.gone:
			case <-time.After(5 * time.Second):
				t.Fatal("the upstream connection stayed open")
			}
			for _, e := range res.Trace {
				if e.Kind == EventTurnEnd && e.Code != "cancelled" {
					t.Errorf("turn_end says %q", e.Code)
				}
			}
		})
	}
}

func TestTimeouts(t *testing.T) {
	t.Run("turn", func(t *testing.T) {
		f := newFake(t, fakeTurn{hang: true})
		b := newBridge(t, f, func(c *Config) { c.Limits.TurnTimeout = 30 * time.Millisecond })
		wantStop(t, runBody(t, b, userBody), StopTurnTimeout)
		<-f.gone
	})
	t.Run("turn stuck inside a call", func(t *testing.T) {
		f := newFake(t, fakeTurn{calls: []fakeCall{{"int_add", `{"a":"1","b":"2"}`}}, hangAfter: 3})
		b := newBridge(t, f, func(c *Config) { c.Limits.TurnTimeout = 30 * time.Millisecond })
		res := runBody(t, b, userBody)
		wantStop(t, res, StopTurnTimeout)
		if res.Calls != 0 {
			t.Errorf("%d calls ran", res.Calls)
		}
	})
	t.Run("request", func(t *testing.T) {
		f := newFake(t, fakeTurn{hang: true})
		b := newBridge(t, f, func(c *Config) { c.Limits.RequestTimeout = 30 * time.Millisecond })
		wantStop(t, runBody(t, b, userBody), StopRequestTimeout)
	})
	t.Run("request during an operation", func(t *testing.T) {
		f := newFake(t, fakeTurn{calls: []fakeCall{{"slow_op", `{}`}}})
		b := newBridge(t, f, func(c *Config) {
			c.Ops = []ops.Op{slowOp(nil, nil)}
			c.Limits.RequestTimeout = 30 * time.Millisecond
		})
		wantStop(t, runBody(t, b, userBody), StopRequestTimeout)
	})
	t.Run("the caller's deadline", func(t *testing.T) {
		f := newFake(t, fakeTurn{hang: true})
		b := newBridge(t, f)
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
		defer cancel()
		res, err := b.Run(ctx, "r", []byte(userBody), Callbacks{})
		if err != nil {
			t.Fatal(err)
		}
		wantStop(t, res, StopRequestTimeout)
	})
}

// The model that never stops calling. It gets exactly MaxTurns completions, and the last one's call
// doesn't run: nothing could read its result.
func TestAModelThatNeverStopsCalling(t *testing.T) {
	f := newFakeFunc(t, func(int, []byte) fakeTurn {
		return fakeTurn{calls: []fakeCall{{"int_add", `{"a":"1","b":"2"}`}}}
	})
	b := newBridge(t, f, func(c *Config) {
		c.Limits.MaxTurns = 5
		c.Limits.MaxCalls = 100
	})
	res := runBody(t, b, userBody)
	wantStop(t, res, StopTurnLimit)
	if f.requestCount() != 5 || res.Turns != 5 || res.Calls != 4 || len(callEnds(res)) != 4 {
		t.Fatalf("%d requests, %d turns, %d calls", f.requestCount(), res.Turns, res.Calls)
	}
}

func TestCallLimits(t *testing.T) {
	call := fakeCall{"int_add", `{"a":"1","b":"2"}`}
	t.Run("per request", func(t *testing.T) {
		f := newFakeFunc(t, func(int, []byte) fakeTurn { return fakeTurn{calls: []fakeCall{call, call}} })
		b := newBridge(t, f, func(c *Config) {
			c.Limits.MaxTurns = 100
			c.Limits.MaxCalls = 5
		})
		res := runBody(t, b, userBody)
		wantStop(t, res, StopCallLimit)
		// Turn 3 would be calls 5 and 6: all of a turn runs or none of it.
		if res.Calls != 4 || f.requestCount() != 3 {
			t.Fatalf("%d calls, %d requests", res.Calls, f.requestCount())
		}
	})
	t.Run("per turn", func(t *testing.T) {
		f := newFake(t, fakeTurn{calls: []fakeCall{call, call, call, call, call}})
		res := runBody(t, newBridge(t, f), userBody)
		wantStop(t, res, StopCallLimit)
		if res.Calls != 0 || len(callEnds(res)) != 0 {
			t.Fatalf("%d calls ran", res.Calls)
		}
	})
	t.Run("per turn, a sparse index", func(t *testing.T) {
		// A huge index must not allocate anything, and counts as one call.
		raw := `data: {"choices":[{"finish_reason":null,"index":0,"delta":{"tool_calls":[{"index":2000000000,"id":"x","type":"function","function":{"name":"int_add","arguments":"{\"a\":\"1\",\"b\":\"2\"}"}}]}}]}` + "\n\n" +
			`data: {"choices":[{"finish_reason":"tool_calls","index":0,"delta":{}}]}` + "\n\ndata: [DONE]\n\n"
		f := newFake(t, fakeTurn{raw: raw}, fakeTurn{content: "3"})
		res := runBody(t, newBridge(t, f), userBody)
		wantStop(t, res, StopCompleted)
		if res.Calls != 1 || f.messages(1)[2]["tool_call_id"] != "t1c0" {
			t.Fatalf("%d calls, %v", res.Calls, f.messages(1))
		}
	})
	t.Run("per turn, non-streaming", func(t *testing.T) {
		var calls []string
		for i := range 5 {
			calls = append(calls, fmt.Sprintf(`{"id":"c%d","type":"function","function":{"name":"int_add","arguments":"{}"}}`, i))
		}
		raw := `{"choices":[{"finish_reason":"tool_calls","index":0,"message":{"role":"assistant","content":"","tool_calls":[` + strings.Join(calls, ",") + `]}}]}`
		f := newFake(t, fakeTurn{raw: raw, rawCT: "application/json"})
		res := runBody(t, newBridge(t, f), userBody)
		wantStop(t, res, StopCallLimit)
		if res.Calls != 0 {
			t.Fatalf("%d calls ran", res.Calls)
		}
	})
}

// A stream that breaks. Whatever arrived of the call doesn't run, even when the arguments look
// complete, and the earlier turns stay in the trace.
func TestAStreamThatBreaks(t *testing.T) {
	call := []fakeCall{{"int_add", `{"a":"1","b":"2"}`}}
	// Turn events: 0 role, 1..6 fragments of the arguments, 7 finish, 8 usage, then [DONE].
	if n := len((&fakeLlama{}).events(fakeTurn{calls: call})); n != 9 {
		t.Fatalf("the script assumes 9 events, the fake sends %d", n)
	}
	for _, c := range []struct {
		name string
		turn fakeTurn
		want StopReason
	}{
		{"before any call", fakeTurn{calls: call, cutAfter: 1}, StopStreamBroken},
		{"inside the arguments", fakeTurn{calls: call, cutAfter: 4}, StopStreamBroken},
		{"inside an event", fakeTurn{calls: call, cutAfter: 4, cutLine: true}, StopStreamBroken},
		{"arguments complete, no finish", fakeTurn{calls: call, cutAfter: 7}, StopStreamBroken},
		{"finish seen, no usage", fakeTurn{calls: call, cutAfter: 8}, StopStreamBroken},
		{"everything but [DONE]", fakeTurn{calls: call, noDone: true}, StopStreamBroken},
		{"answer, no [DONE]", fakeTurn{content: "42", noDone: true}, StopStreamBroken},
		{"not even a header", fakeTurn{raw: "garbage\n\n"}, StopBadUpstream},
		{"an event that isn't JSON", fakeTurn{raw: "data: {nope}\n\n"}, StopBadUpstream},
		{"[DONE] without a finish", fakeTurn{raw: "data: [DONE]\n\n"}, StopBadUpstream},
		{"a second choice", fakeTurn{raw: `data: {"choices":[{"index":1,"delta":{"content":"x"}}]}` + "\n\n"}, StopBadUpstream},
		{"a call changes its name", fakeTurn{raw: `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"a"}}]}}]}` + "\n\n" +
			`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"b"}}]}}]}` + "\n\n"}, StopBadUpstream},
		{"tool_calls without calls", fakeTurn{finish: "tool_calls"}, StopBadUpstream},
		{"calls with finish stop", fakeTurn{calls: call, finish: "stop"}, StopBadUpstream},
		{"an unknown finish", fakeTurn{content: "x", finish: "content_filter"}, StopBadUpstream},
		{"wrong content type", fakeTurn{raw: "<html>", rawCT: "text/html"}, StopBadUpstream},
		{"non-streaming, not JSON", fakeTurn{raw: "{", rawCT: "application/json"}, StopBadUpstream},
		{"non-streaming, no choice", fakeTurn{raw: `{"choices":[]}`, rawCT: "application/json"}, StopBadUpstream},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := newFake(t, fakeTurn{calls: []fakeCall{{"int_add", `{"a":"5","b":"5"}`}}}, c.turn)
			res := runBody(t, newBridge(t, f), userBody)
			wantStop(t, res, c.want)
			// The first turn's call ran, the broken turn's didn't.
			if res.Calls != 1 || len(callEnds(res)) != 1 || res.Turns != 2 {
				t.Fatalf("%d calls, %d turns", res.Calls, res.Turns)
			}
			if strings.Contains(fmt.Sprint(res.Trace), "t2c0") {
				t.Fatal("the broken turn's call shows up in the trace")
			}
		})
	}
}

func TestAnAnswerCutOrWithAnUnparsedCall(t *testing.T) {
	t.Run("out of tokens", func(t *testing.T) {
		f := newFake(t, fakeTurn{content: "The answer is", finish: "length"})
		res := runBody(t, newBridge(t, f), userBody)
		wantStop(t, res, StopTruncated)
		if res.Content != "The answer is" {
			t.Errorf("content %q", res.Content)
		}
	})
	t.Run("a marker the server didn't parse", func(t *testing.T) {
		// What llama-server does with a <tool_call> it can't read: the tiny checkpoint, forced to
		// repeat the token, gets it back as plain content.
		f := newFake(t, fakeTurn{content: `<tool_call>{"name":"int_add"`})
		res := runBody(t, newBridge(t, f), userBody)
		wantStop(t, res, StopUnparsedCall)
		if res.Calls != 0 {
			t.Errorf("%d calls ran", res.Calls)
		}
	})
	t.Run("the marker in the reasoning is fine", func(t *testing.T) {
		f := newFake(t, fakeTurn{reasoning: "I could use <tool_call>.", content: "No need."})
		wantStop(t, runBody(t, newBridge(t, f), userBody), StopCompleted)
	})
	t.Run("another model's marker", func(t *testing.T) {
		f := newFake(t, fakeTurn{content: "[TOOL_CALLS]"})
		res := runBody(t, newBridge(t, f, func(c *Config) { c.CallMarker = "[TOOL_CALLS]" }), userBody)
		wantStop(t, res, StopUnparsedCall)
	})
}

func TestUpstreamProblems(t *testing.T) {
	t.Run("no server", func(t *testing.T) {
		f := newFake(t)
		b := newBridge(t, f, func(c *Config) { c.Upstream = "http://127.0.0.1:1" })
		wantStop(t, runBody(t, b, userBody), StopUnreachable)
	})
	t.Run("a redirect is not followed", func(t *testing.T) {
		var elsewhere atomic.Int32
		other := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { elsewhere.Add(1) }))
		defer other.Close()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, other.URL, http.StatusTemporaryRedirect)
		}))
		defer srv.Close()
		f := newFake(t)
		b := newBridge(t, f, func(c *Config) { c.Upstream = srv.URL })
		res := runBody(t, b, userBody)
		wantStop(t, res, StopUpstreamHTTP)
		if elsewhere.Load() != 0 || res.Failure.HTTPStatus != http.StatusTemporaryRedirect {
			t.Fatalf("the prompt went elsewhere: %+v", res.Failure)
		}
	})
	t.Run("a response past the limit", func(t *testing.T) {
		for _, c := range []struct {
			name string
			turn fakeTurn
		}{
			{"many events", fakeTurn{content: strings.Repeat("x", 4000)}},
			{"one long event", fakeTurn{raw: "data: " + strings.Repeat("x", 4000) + "\n\n"}},
			{"non-streaming", fakeTurn{raw: strings.Repeat(" ", 4000), rawCT: "application/json"}},
		} {
			t.Run(c.name, func(t *testing.T) {
				f := newFake(t, c.turn)
				b := newBridge(t, f, func(cfg *Config) { cfg.Limits.MaxTurnBytes = 2000 })
				wantStop(t, runBody(t, b, userBody), StopResponseTooLarge)
			})
		}
	})
}

func TestClientRequestsThatAreRefused(t *testing.T) {
	f := newFake(t)
	b := newBridge(t, f)
	for _, body := range []string{
		``, `[]`, `{}`, `{"messages":[]}`, `{"messages":"hi"}`, `{"messages":[{"content":"no role"}]}`, `{"messages":[1]}`,
		`{"messages":[{"role":"user","content":"x"}],"stream":"yes"}`,
		`{"messages":[{"role":"user","content":"x"}],"tools":[]}`,
		`{"messages":[{"role":"user","content":"x"}],"functions":[]}`,
		`{"messages":[{"role":"user","content":"x"}],"grammar":"root ::= \"a\""}`,
		`{"messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_object"}}`,
		`{"messages":[{"role":"user","content":"x"}],"tool_choice":"required"}`,
		`{"messages":[{"role":"user","content":"x"}],"tool_choice":"none"}`,
		`{"messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"function"}}`,
		`{"messages":[{"role":"user","content":"x"}],"n":2}`,
		`{"messages":[{"role":"user","content":"x"}],"max_tokens":0}`,
		`{"messages":[{"role":"user","content":"x"}],"max_tokens":999999}`,
		`{"messages":[{"role":"user","content":"x"}],"max_tokens":"9"}`,
		`{"messages":[{"role":"user","content":"x"}],"n_predict":-1}`,
		`{"messages":[{"role":"user","content":"x"}],"max_completion_tokens":5}`,
		`{"messages":[{"role":"user","content":"x"}],"stream_options":{"include_usage":false}}`,
	} {
		_, err := b.Run(context.Background(), "", []byte(body), Callbacks{})
		var refused *RequestError
		if !errors.As(err, &refused) || refused.Status != http.StatusBadRequest {
			t.Errorf("%s: %v", body, err)
		}
	}
	if _, err := b.Run(context.Background(), "bad id!", []byte(userBody), Callbacks{}); err == nil {
		t.Error("a bad request ID passes")
	}
	if f.requestCount() != 0 {
		t.Errorf("%d requests reached the server", f.requestCount())
	}
}

func TestAConfigWithoutLimitsIsRefused(t *testing.T) {
	if _, err := New(Config{Upstream: "http://x", OpLimits: ops.DefaultLimits()}); err == nil {
		t.Error("zero limits pass")
	}
	if _, err := New(Config{Limits: DefaultLimits(), OpLimits: ops.DefaultLimits()}); err == nil {
		t.Error("no upstream passes")
	}
	if _, err := New(Config{Upstream: "http://x", Limits: DefaultLimits()}); err == nil {
		t.Error("zero operation limits pass")
	}
	if _, err := New(Config{Upstream: "http://x", Limits: DefaultLimits(), OpLimits: ops.DefaultLimits(), Allow: []string{"typo"}}); err == nil {
		t.Error("an allowlist with a typo passes")
	}
}

// Two runs of one request against one script send the same bytes, though the server's call IDs are
// random, and give the same canonical trace and the same messages. Operations run concurrently, a
// third of them failing, and neither changes that.
func TestReproducible(t *testing.T) {
	script := func() *fakeLlama {
		return newFake(t,
			fakeTurn{reasoning: "plan", content: "first", calls: []fakeCall{
				{"int_mul", `{"a":"987654321987654321","b":"123456789123456789"}`},
				{"hash_sha256", `{"data":"abc"}`},
				{"int_div", `{"a":"1","b":"0"}`},
				{"array_sort", `{"values":["3","-1","2"]}`},
			}},
			fakeTurn{calls: []fakeCall{{"no_such_op", `{"x":1}`}, {"int_add", `not json`}}},
			fakeTurn{content: "The answer."})
	}
	var first *Result
	var firstRequests [][]byte
	for i := range 8 {
		f := script()
		res := runBody(t, newBridge(t, f), userBody)
		wantStop(t, res, StopCompleted)
		requests := [][]byte{f.request(0), f.request(1), f.request(2)}
		if first == nil {
			first, firstRequests = res, requests
			continue
		}
		for turn := range requests {
			if string(requests[turn]) != string(firstRequests[turn]) {
				t.Fatalf("run %d, turn %d:\n%s\n%s", i, turn, requests[turn], firstRequests[turn])
			}
		}
		if !reflect.DeepEqual(Canonical(res.Trace), Canonical(first.Trace)) {
			t.Fatalf("run %d: the traces differ:\n%+v\n%+v", i, Canonical(res.Trace), Canonical(first.Trace))
		}
		if !reflect.DeepEqual(res.Messages, first.Messages) || res.Content != first.Content {
			t.Fatalf("run %d: the messages differ", i)
		}
	}
	// The server's random IDs are in the raw trace and gone from the canonical one.
	var saw bool
	for _, e := range first.Trace {
		saw = saw || e.UpstreamID != ""
	}
	if !saw {
		t.Error("the trace lost the server's call IDs")
	}
	for _, e := range Canonical(first.Trace) {
		if e.UpstreamID != "" || e.ElapsedMS != 0 || e.Usage != nil {
			t.Fatalf("not canonical: %+v", e)
		}
	}
}

// The scheduler's limit on open requests, and a request ID in use.
func TestAdmission(t *testing.T) {
	f := newFakeFunc(t, func(int, []byte) fakeTurn { return fakeTurn{hang: true} })
	b := newBridge(t, f, func(c *Config) { c.Limits.MaxRequests = 2 })
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var wg sync.WaitGroup
	for _, id := range []string{"one", "two"} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res, err := b.Run(ctx, id, []byte(userBody), Callbacks{})
			if err != nil {
				t.Error(err)
				return
			}
			wantStop(t, res, StopCancelled)
		}()
	}
	waitFor(t, func() bool { return f.requestCount() == 2 })
	var refused *RequestError
	if _, err := b.Run(context.Background(), "one", []byte(userBody), Callbacks{}); !errors.As(err, &refused) || refused.Status != http.StatusConflict {
		t.Errorf("a duplicate ID: %v", err)
	}
	if _, err := b.Run(context.Background(), "three", []byte(userBody), Callbacks{}); !errors.As(err, &refused) || refused.Status != http.StatusTooManyRequests {
		t.Errorf("a third request: %v", err)
	}
	cancel()
	wg.Wait()
	// Both slots are free again, and an ID can be used again.
	g := newFake(t, fakeTurn{content: "x"}, fakeTurn{content: "y"})
	b2 := newBridge(t, g)
	wantStop(t, runBody(t, b2, userBody), StopCompleted)
	wantStop(t, runBody(t, b2, userBody), StopCompleted)
}

func TestManyRequestsAtOnce(t *testing.T) {
	f := newFakeFunc(t, func(_ int, body []byte) fakeTurn {
		if strings.Contains(string(body), `"role":"tool"`) {
			return fakeTurn{content: "done"}
		}
		return fakeTurn{calls: []fakeCall{{"int_add", `{"a":"1","b":"2"}`}, {"int_mul", `{"a":"3","b":"4"}`}}}
	})
	b := newBridge(t, f, func(c *Config) {
		c.Limits.MaxRequests = 16
		c.Limits.MaxRunningOps = 2
	})
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res, err := b.Run(context.Background(), "", []byte(userBody), Callbacks{})
			if err != nil {
				t.Error(err)
				return
			}
			wantStop(t, res, StopCompleted)
			if ends := callEnds(res); len(ends) != 2 || !strings.Contains(ends[0].Result, `"value":"3"`) || !strings.Contains(ends[1].Result, `"value":"12"`) {
				t.Errorf("results %+v", ends)
			}
		}()
	}
	wg.Wait()
}

func TestClosingTheBridgeEndsRequests(t *testing.T) {
	started := make(chan struct{}, 1)
	f := newFake(t, fakeTurn{calls: []fakeCall{{"slow_op", `{}`}}})
	cfg := Config{Upstream: f.srv.URL, Limits: testLimits(), OpLimits: ops.DefaultLimits(), Ops: []ops.Op{slowOp(started, nil)}}
	b, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	results := make(chan *Result, 1)
	go func() {
		res, err := b.Run(context.Background(), "r", []byte(userBody), Callbacks{})
		if err != nil {
			t.Error(err)
		}
		results <- res
	}()
	<-started
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := b.Close(ctx); err != nil {
		t.Fatal(err)
	}
	wantStop(t, <-results, StopCancelled)
	if _, err := b.Run(context.Background(), "r2", []byte(userBody), Callbacks{}); err == nil {
		t.Error("a closed bridge takes requests")
	}
}

func TestTraceSinkSeesEveryEvent(t *testing.T) {
	var mu sync.Mutex
	var got []Event
	f := newFake(t, fakeTurn{calls: []fakeCall{{"int_add", `{"a":"1","b":"2"}`}}}, fakeTurn{content: "3"})
	b := newBridge(t, f, func(c *Config) {
		c.Trace = func(e Event) {
			mu.Lock()
			defer mu.Unlock()
			got = append(got, e)
		}
	})
	res := runBody(t, b, userBody)
	mu.Lock()
	defer mu.Unlock()
	if !reflect.DeepEqual(got, res.Trace) {
		t.Errorf("sink %+v\nresult %+v", got, res.Trace)
	}
}

func TestTraceClipsLongText(t *testing.T) {
	args := fmt.Sprintf(`{"data":"%s"}`, strings.Repeat("é", 4000))
	f := newFake(t, fakeTurn{calls: []fakeCall{{"hash_sha256", args}}}, fakeTurn{content: "done"})
	res := runBody(t, newBridge(t, f), userBody)
	wantStop(t, res, StopCompleted)
	for _, e := range res.Trace {
		if e.Kind != EventCallStart {
			continue
		}
		if !strings.HasSuffix(e.Args, "...") || len(e.Args) > traceTextLimit+3 || strings.ContainsRune(e.Args, '\uFFFD') {
			t.Errorf("clipped arguments of %d bytes", len(e.Args))
		}
		if e.ArgsSHA256 != sha256Hex([]byte(args)) {
			t.Error("the hash doesn't cover the whole arguments")
		}
	}
}

// --- HTTP API

func serveBridge(t *testing.T, b *Bridge) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(b.Handler())
	t.Cleanup(srv.Close)
	return srv
}

func post(t *testing.T, url, body string, headers ...string) (*http.Response, []byte) {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i+1 < len(headers); i += 2 {
		req.Header.Set(headers[i], headers[i+1])
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return resp, raw
}

func TestHTTPCompletion(t *testing.T) {
	f := newFake(t, fakeTurn{calls: []fakeCall{{"int_add", `{"a":"40","b":"2"}`}}}, fakeTurn{reasoning: "easy", content: "42"})
	srv := serveBridge(t, newBridge(t, f))
	resp, raw := post(t, srv.URL+"/v1/chat/completions", userBody, "X-Request-Id", "my-id")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("%d: %s", resp.StatusCode, raw)
	}
	var out struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Role, Content    string
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
		Bridge Result `json:"kolibri_bridge"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	if out.ID != "my-id" || out.Object != "chat.completion" || out.Choices[0].Message.Content != "42" ||
		out.Choices[0].Message.ReasoningContent != "easy" || out.Choices[0].FinishReason != "stop" || out.Usage.TotalTokens != 30 {
		t.Errorf("%s", raw)
	}
	if out.Bridge.Stop != StopCompleted || out.Bridge.Protocol != ops.Version || out.Bridge.Turns != 2 || out.Bridge.Calls != 1 || len(out.Bridge.Trace) == 0 {
		t.Errorf("%+v", out.Bridge)
	}
}

func TestHTTPFailures(t *testing.T) {
	for _, c := range []struct {
		name   string
		turn   fakeTurn
		status int
		kind   StopReason
	}{
		{"truncated", fakeTurn{content: "x", finish: "length"}, http.StatusUnprocessableEntity, StopTruncated},
		{"broken", fakeTurn{content: "x", noDone: true}, http.StatusBadGateway, StopStreamBroken},
		{"refused", fakeTurn{raw: `{"error":{"code":400,"message":"m","type":"exceed_context_size_error"}}`, rawCT: "application/json", status: 400}, http.StatusBadGateway, StopUpstreamHTTP},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := newFake(t, c.turn)
			srv := serveBridge(t, newBridge(t, f))
			resp, raw := post(t, srv.URL+"/v1/chat/completions", userBody)
			if resp.StatusCode != c.status {
				t.Fatalf("%d: %s", resp.StatusCode, raw)
			}
			var out struct {
				Error struct {
					Code int    `json:"code"`
					Type string `json:"type"`
				} `json:"error"`
				Bridge Result `json:"kolibri_bridge"`
			}
			if err := json.Unmarshal(raw, &out); err != nil {
				t.Fatal(err)
			}
			if out.Error.Code != c.status || out.Error.Type != string(c.kind) || out.Bridge.Stop != c.kind || len(out.Bridge.Trace) == 0 {
				t.Fatalf("%s", raw)
			}
		})
	}
}

func TestHTTPRefusals(t *testing.T) {
	f := newFake(t)
	srv := serveBridge(t, newBridge(t, f, func(c *Config) { c.Limits.MaxRequestBytes = 1000 }))
	for _, c := range []struct {
		body   string
		status int
	}{
		{`{"messages":[]}`, http.StatusBadRequest},
		{`{"messages":[{"role":"user","content":"x"}],"tools":[]}`, http.StatusBadRequest},
		{`{"messages":[{"role":"user","content":"x"}],"tools":[],"stream":true}`, http.StatusBadRequest},
		{`{"padding":"` + strings.Repeat("x", 2000) + `"}`, http.StatusRequestEntityTooLarge},
	} {
		if resp, raw := post(t, srv.URL+"/v1/chat/completions", c.body); resp.StatusCode != c.status {
			t.Errorf("%.60s: %d: %s", c.body, resp.StatusCode, raw)
		}
	}
	if resp, raw := post(t, srv.URL+"/v1/chat/completions", userBody, "X-Request-Id", "bad id"); resp.StatusCode != http.StatusBadRequest {
		t.Errorf("bad ID: %d: %s", resp.StatusCode, raw)
	}
	if resp, raw := post(t, srv.URL+"/v1/chat/completions", `{"messages":[{"role":"user","content":"x"}],"stream":true}`, "X-Request-Id", "bad id"); resp.StatusCode != http.StatusBadRequest {
		t.Errorf("bad ID, stream: %d: %s", resp.StatusCode, raw)
	}
}

// readEvents splits a stream into unnamed data and kolibri_bridge events.
func readEvents(t *testing.T, r io.Reader) (data []string, trace []Event) {
	t.Helper()
	scanner := bufio.NewScanner(r)
	scanner.Buffer(nil, 1<<20)
	named := ""
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			named = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			text := strings.TrimPrefix(line, "data: ")
			if named == "kolibri_bridge" {
				var e Event
				if err := json.Unmarshal([]byte(text), &e); err != nil {
					t.Fatal(err)
				}
				trace = append(trace, e)
			} else {
				data = append(data, text)
			}
			named = ""
		}
	}
	return data, trace
}

func TestHTTPStreaming(t *testing.T) {
	f := newFake(t,
		fakeTurn{reasoning: "think", content: "Let me add.", calls: []fakeCall{{"int_add", `{"a":"40","b":"2"}`}}},
		fakeTurn{content: "It is 42."})
	srv := serveBridge(t, newBridge(t, f))
	resp, raw := post(t, srv.URL+"/v1/chat/completions", `{"messages":[{"role":"user","content":"x"}],"stream":true}`, "X-Request-Id", "s1")
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("%d %s", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	data, trace := readEvents(t, strings.NewReader(string(raw)))
	if data[len(data)-1] != "[DONE]" {
		t.Fatalf("no [DONE]: %v", data)
	}
	var content, reasoning, finish strings.Builder
	for _, d := range data[:len(data)-1] {
		var c struct {
			ID      string `json:"id"`
			Choices []struct {
				FinishReason *string `json:"finish_reason"`
				Delta        struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(d), &c); err != nil {
			t.Fatal(err)
		}
		if c.ID != "s1" {
			t.Errorf("chunk ID %q", c.ID)
		}
		content.WriteString(c.Choices[0].Delta.Content)
		reasoning.WriteString(c.Choices[0].Delta.ReasoningContent)
		if c.Choices[0].FinishReason != nil {
			finish.WriteString(*c.Choices[0].FinishReason)
		}
	}
	if content.String() != "Let me add.It is 42." || reasoning.String() != "think" || finish.String() != "stop" {
		t.Errorf("content %q, reasoning %q, finish %q", content.String(), reasoning.String(), finish.String())
	}
	if len(trace) == 0 || trace[0].Kind != EventRequestStart || trace[len(trace)-1].Stop != StopCompleted {
		t.Fatalf("trace %+v", trace)
	}
}

// A failure after the stream opened is an error event and no [DONE], as llama-server does it.
func TestHTTPStreamingFailure(t *testing.T) {
	f := newFake(t, fakeTurn{content: "partial", cutAfter: 2})
	srv := serveBridge(t, newBridge(t, f))
	resp, raw := post(t, srv.URL+"/v1/chat/completions", `{"messages":[{"role":"user","content":"x"}],"stream":true}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("%d", resp.StatusCode)
	}
	data, trace := readEvents(t, strings.NewReader(string(raw)))
	last := data[len(data)-1]
	if !strings.Contains(last, `"type":"stream_broken"`) || strings.Contains(string(raw), "[DONE]") {
		t.Fatalf("%s", raw)
	}
	if trace[len(trace)-1].Stop != StopStreamBroken {
		t.Fatalf("trace %+v", trace)
	}
}

// A client that leaves cancels the request and its upstream connection.
func TestHTTPClientLeaves(t *testing.T) {
	var ended sync.WaitGroup
	ended.Add(1)
	var final atomic.Value
	f := newFake(t, fakeTurn{hang: true})
	b := newBridge(t, f, func(c *Config) {
		c.Trace = func(e Event) {
			if e.Kind == EventRequestEnd {
				final.Store(e)
				ended.Done()
			}
		}
	})
	srv := serveBridge(t, b)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL+"/v1/chat/completions", strings.NewReader(userBody))
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		waitFor(t, func() bool { return f.requestCount() == 1 })
		cancel()
	}()
	if _, err := http.DefaultClient.Do(req); err == nil {
		t.Error("the client got an answer")
	}
	ended.Wait()
	<-f.gone
	if e := final.Load().(Event); e.Stop != StopCancelled {
		t.Errorf("end %+v", e)
	}
}

func TestHTTPCancel(t *testing.T) {
	started := make(chan struct{}, 2)
	f := newFakeFunc(t, func(int, []byte) fakeTurn { return fakeTurn{calls: []fakeCall{{"slow_op", `{}`}}} })
	b := newBridge(t, f, func(c *Config) { c.Ops = []ops.Op{slowOp(started, nil)} })
	srv := serveBridge(t, b)

	done := make(chan []byte, 1)
	go func() {
		_, raw := post(t, srv.URL+"/v1/chat/completions", userBody, "X-Request-Id", "c1")
		done <- raw
	}()
	<-started
	// One call: the request goes on to its next turn, which starts another.
	if resp, raw := post(t, srv.URL+"/v1/executions/c1/cancel", `{"call_id":"t1c0"}`); resp.StatusCode != http.StatusOK {
		t.Fatalf("%d: %s", resp.StatusCode, raw)
	}
	<-started
	if resp, raw := post(t, srv.URL+"/v1/executions/c1/cancel", `{"call_id":"nope"}`); resp.StatusCode != http.StatusNotFound {
		t.Errorf("unknown call: %d: %s", resp.StatusCode, raw)
	}
	if resp, raw := post(t, srv.URL+"/v1/executions/c1/cancel", `{bad`); resp.StatusCode != http.StatusBadRequest {
		t.Errorf("bad body: %d: %s", resp.StatusCode, raw)
	}
	// The whole request.
	if resp, raw := post(t, srv.URL+"/v1/executions/c1/cancel", ``); resp.StatusCode != http.StatusOK {
		t.Fatalf("%d: %s", resp.StatusCode, raw)
	}
	raw := <-done
	var out struct {
		Error struct {
			Code int    `json:"code"`
			Type string `json:"type"`
		} `json:"error"`
		Bridge Result `json:"kolibri_bridge"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("%v: %s", err, raw)
	}
	if out.Error.Code != 499 || out.Error.Type != "cancelled" || out.Bridge.Stop != StopCancelled {
		t.Fatalf("%s", raw)
	}
	ends := callEnds(&out.Bridge)
	if len(ends) != 1 || ends[0].Code != "cancelled" {
		t.Fatalf("results %+v", ends)
	}
	if resp, raw := post(t, srv.URL+"/v1/executions/c1/cancel", ``); resp.StatusCode != http.StatusNotFound {
		t.Errorf("after the end: %d: %s", resp.StatusCode, raw)
	}
}

func TestHTTPManifestAndHealth(t *testing.T) {
	f := newFake(t)
	srv := serveBridge(t, newBridge(t, f, func(c *Config) { c.Allow = []string{"int_add"} }))
	resp, err := http.Get(srv.URL + "/v1/bridge/tools")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Protocol int `json:"protocol"`
		Limits   map[string]float64
		Ops      []struct {
			Name   string
			Input  map[string]any
			Output map[string]any
		} `json:"ops"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Protocol != ops.Version || len(out.Ops) != 1 || out.Ops[0].Name != "int_add" || out.Ops[0].Output["type"] != "object" {
		t.Errorf("%+v", out)
	}
	if out.Limits["max_turns"] == 0 || out.Limits["max_int_bits"] == 0 {
		t.Errorf("limits %v", out.Limits)
	}
	health, err := http.Get(srv.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer health.Body.Close()
	if health.StatusCode != http.StatusOK {
		t.Errorf("health %d", health.StatusCode)
	}
}
