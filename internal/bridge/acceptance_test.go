package bridge

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"math/big"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"kolibri-llm/internal/bridge/ops"
)

// The acceptance task, written for issue #59's criterion: the issue names no task. Three steps, each
// needing the last one's exact output. A model can't do the last step in its head, so a correct
// digest in the answer means the results went through it. The expected values are literals, computed
// outside this repo's Go code.
const (
	acceptA       = "987654321987654321"
	acceptB       = "123456789123456789"
	acceptProduct = "121932631356500531347203169112635269"
	acceptSum     = "121932631356500531347203169112635270"
	acceptDigest  = "8794da494318df92e876854b96a3ed629fa76ad21764a7562a1b6b5bbb76428f"

	// sha256 of the chat_template in tokenizer_config.json of the pinned revision, the same constant
	// as tools/chat/check_chat.py. It tells the released template from another one. It doesn't identify
	// the weights: random weights under another file name get past it.
	kolibriTemplateSHA256 = "9ba35d4bd6baa26b66aa75d03a922dfee98b16bb1fa37481b195d247267b0f97"
)

var acceptPrompt = "Work through these steps with the tools. Do the arithmetic only with the tools, and copy every number digit for digit.\n" +
	"1. Multiply " + acceptA + " by " + acceptB + ".\n" +
	"2. Add 1 to the product.\n" +
	"3. Take the SHA-256 of the decimal digits of that sum.\n" +
	"Call one tool at a time and wait for its result. When you have the digest, answer with the 64 hex digits and nothing else."

// The oracle isn't the code under test: this checks the literals with math/big and crypto/sha256
// directly, and it runs on every test run, unlike the task.
func TestAcceptanceOracle(t *testing.T) {
	a, _ := new(big.Int).SetString(acceptA, 10)
	b, _ := new(big.Int).SetString(acceptB, 10)
	product := new(big.Int).Mul(a, b)
	sum := new(big.Int).Add(product, big.NewInt(1))
	digest := sha256.Sum256([]byte(sum.String()))
	if product.String() != acceptProduct || sum.String() != acceptSum || hex.EncodeToString(digest[:]) != acceptDigest {
		t.Fatalf("%s %s %x", product, sum, digest)
	}
}

type acceptStep struct {
	op     string
	check  func(args map[string]string) bool
	result string
}

var acceptSteps = []acceptStep{
	{"int_mul", func(args map[string]string) bool {
		return (args["a"] == acceptA && args["b"] == acceptB) || (args["a"] == acceptB && args["b"] == acceptA)
	}, acceptProduct},
	{"int_add", func(args map[string]string) bool {
		return (args["a"] == acceptProduct && args["b"] == "1") || (args["a"] == "1" && args["b"] == acceptProduct)
	}, acceptSum},
	{"hash_sha256", func(args map[string]string) bool {
		return args["data"] == acceptSum && (args["encoding"] == "" || args["encoding"] == "utf8")
	}, acceptDigest},
}

// checkAcceptance judges one run exactly, and returns nothing a human has to interpret: the stop
// reason, each step's operation, inputs and output, in order, and the answer.
func checkAcceptance(res *Result) []string {
	var problems []string
	fail := func(s string) { problems = append(problems, s) }
	if res.Stop != StopCompleted {
		fail("stop reason " + string(res.Stop))
	}
	args := map[string]string{}
	step := 0
	for _, e := range res.Trace {
		switch e.Kind {
		case EventCallStart:
			args[e.CallID] = e.Args
		case EventCallEnd:
			if e.Code != "" {
				// The model may get a call wrong and repair it. That isn't the task failing; it is
				// reported, and the steps below still have to happen.
				continue
			}
			if step == len(acceptSteps) {
				fail("a call after the last step: " + e.Op + " " + args[e.CallID])
				continue
			}
			want := acceptSteps[step]
			var parsed map[string]string
			if err := json.Unmarshal([]byte(args[e.CallID]), &parsed); err != nil {
				fail("step " + strconv.Itoa(step+1) + ": arguments " + args[e.CallID] + ": " + err.Error())
				continue
			}
			if e.Op != want.op || !want.check(parsed) {
				fail("step " + strconv.Itoa(step+1) + ": got " + e.Op + " " + args[e.CallID] + ", want " + want.op)
				continue
			}
			if !strings.Contains(e.Result, `"`+want.result+`"`) {
				fail("step " + strconv.Itoa(step+1) + ": result " + e.Result + ", want " + want.result)
			}
			step++
		}
	}
	if step != len(acceptSteps) {
		fail("the model reached " + strconv.Itoa(step) + " of " + strconv.Itoa(len(acceptSteps)) + " steps")
	}
	digests := regexp.MustCompile(`\b[0-9a-f]{64}\b`).FindAllString(res.Content, -1)
	if len(digests) != 1 || digests[0] != acceptDigest {
		fail("the answer doesn't hold exactly the digest: " + strconv.Quote(clipText(res.Content)))
	}
	return problems
}

// The judge itself has to fail, or the real run proves nothing: a fake that plays the task right
// passes it, and one wrong in any way doesn't.
func TestAcceptanceJudge(t *testing.T) {
	good := []fakeTurn{
		{reasoning: "Multiply.", calls: []fakeCall{{"int_mul", `{"a":"` + acceptA + `","b":"` + acceptB + `"}`}}},
		{calls: []fakeCall{{"int_add", `{"a":"` + acceptProduct + `","b":"1"}`}}},
		{calls: []fakeCall{{"hash_sha256", `{"data":"` + acceptSum + `"}`}}},
		{content: acceptDigest},
	}
	replace := func(turn int, with fakeTurn) []fakeTurn {
		out := append([]fakeTurn(nil), good...)
		out[turn] = with
		return out
	}
	for _, c := range []struct {
		name string
		turn []fakeTurn
		ok   bool
	}{
		{"right", good, true},
		{"a repaired mistake", append([]fakeTurn{{calls: []fakeCall{{"int_mul", `{"a":1.5,"b":"2"}`}}}}, good...), true},
		{"a wrong digest", replace(3, fakeTurn{content: strings.Repeat("0", 64)}), false},
		{"two digests", replace(3, fakeTurn{content: acceptDigest + " " + strings.Repeat("0", 64)}), false},
		{"no digest", replace(3, fakeTurn{content: "done"}), false},
		{"a miscopied number", replace(1, fakeTurn{calls: []fakeCall{{"int_add", `{"a":"121932631356500531347203169112635268","b":"1"}`}}}), false},
		{"a skipped step", []fakeTurn{good[0], good[2], good[3]}, false},
		{"no tool at all", []fakeTurn{{content: acceptDigest}}, false},
		{"the wrong tool", replace(0, fakeTurn{calls: []fakeCall{{"int_add", `{"a":"` + acceptA + `","b":"` + acceptB + `"}`}}}), false},
		{"steps out of order", []fakeTurn{good[1], good[0], good[2], good[3]}, false},
		{"a truncated end", replace(3, fakeTurn{content: acceptDigest, finish: "length"}), false},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := newFake(t, c.turn...)
			b := newBridge(t, f, func(cfg *Config) { cfg.Limits.MaxTurns = 8 })
			problems := checkAcceptance(runBody(t, b, acceptBody(t, 7, "low")))
			if (len(problems) == 0) != c.ok {
				t.Fatalf("ok %v, problems: %q", c.ok, problems)
			}
		})
	}
}

func acceptBody(t *testing.T, seed int, effort string) string {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"messages":         []map[string]string{{"role": "user", "content": acceptPrompt}},
		"temperature":      0,
		"seed":             seed,
		"max_tokens":       2048,
		"reasoning_effort": effort,
	})
	if err != nil {
		t.Fatal(err)
	}
	return string(body)
}

func acceptanceAllow() []string { return []string{"hash_sha256", "int_add", "int_mul", "int_sub"} }

// checkServerIsKolibri refuses to judge anything but the released template on real weights.
func checkServerIsKolibri(t *testing.T, url string) {
	t.Helper()
	resp, err := http.Get(strings.TrimSuffix(url, "/") + "/props")
	if err != nil {
		t.Fatalf("no server at %s: %v", url, err)
	}
	defer resp.Body.Close()
	var props struct {
		ChatTemplate string `json:"chat_template"`
		ModelPath    string `json:"model_path"`
		Settings     struct {
			NCtx int `json:"n_ctx"`
		} `json:"default_generation_settings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&props); err != nil {
		t.Fatal(err)
	}
	if got := sha256Hex([]byte(props.ChatTemplate)); got != kolibriTemplateSHA256 {
		t.Fatalf("the server's template is %s, not Kolibri's released %s", got, kolibriTemplateSHA256)
	}
	if strings.Contains(filepath.Base(props.ModelPath), "tiny") {
		t.Fatalf("%s is the tiny checkpoint: this test is for real weights", props.ModelPath)
	}
	t.Logf("server: %s, n_ctx %d", props.ModelPath, props.Settings.NCtx)
}

// TestAcceptanceRealModel tests issue #59's acceptance criterion on one task: unmodified Kolibri
// completes a deterministic multi-step task through the bridge, checked exactly. It needs a real
// weight in a running llama-server, which is why it is skipped by default. See
// docs/execution-bridge.md for the commands and for the one real run so far.
//
//	KOLIBRI_BRIDGE_ACCEPT_URL   the server, for example http://127.0.0.1:8099; unset: skip
//	KOLIBRI_BRIDGE_ACCEPT_RUNS  runs of the task, default 1; above 1 they must give one trace
//	KOLIBRI_BRIDGE_ACCEPT_EFFORT  reasoning_effort, default low
//	KOLIBRI_BRIDGE_ACCEPT_TRACE   a file for the trace of each run, as JSON lines
func TestAcceptanceRealModel(t *testing.T) {
	url := os.Getenv("KOLIBRI_BRIDGE_ACCEPT_URL")
	if url == "" {
		t.Skip("KOLIBRI_BRIDGE_ACCEPT_URL is not set: this runs on real weights, see docs/execution-bridge.md")
	}
	checkServerIsKolibri(t, url)

	runs := 1
	if s := os.Getenv("KOLIBRI_BRIDGE_ACCEPT_RUNS"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil || n < 1 {
			t.Fatalf("KOLIBRI_BRIDGE_ACCEPT_RUNS %q", s)
		}
		runs = n
	}
	effort := os.Getenv("KOLIBRI_BRIDGE_ACCEPT_EFFORT")
	if effort == "" {
		effort = "low"
	}
	var traceOut io.Writer = io.Discard
	if path := os.Getenv("KOLIBRI_BRIDGE_ACCEPT_TRACE"); path != "" {
		file, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		traceOut = file
	}

	lim := DefaultLimits()
	lim.TurnTimeout = 30 * time.Minute
	lim.RequestTimeout = 2 * time.Hour
	b, err := New(Config{Upstream: url, Limits: lim, OpLimits: ops.DefaultLimits(), Allow: acceptanceAllow()})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		_ = b.Close(ctx)
	}()

	encoder := json.NewEncoder(traceOut)
	var first *Result
	for i := range runs {
		res, err := b.Run(context.Background(), "accept", []byte(acceptBody(t, 7, effort)), Callbacks{})
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range res.Trace {
			if err := encoder.Encode(e); err != nil {
				t.Fatal(err)
			}
			switch e.Kind {
			case EventTurnEnd:
				t.Logf("run %d turn %d: finish %s, %d calls, reasoning %q, content %q, usage %+v", i+1, e.Turn, e.Finish, e.Calls, e.Reasoning, e.Content, e.Usage)
			case EventCallStart:
				t.Logf("run %d call %s: %s %s", i+1, e.CallID, e.Op, e.Args)
			case EventCallEnd:
				t.Logf("run %d result %s: %s", i+1, e.CallID, e.Result)
			}
		}
		t.Logf("run %d: stop %s, %d turns, %d calls, usage %+v, answer %q", i+1, res.Stop, res.Turns, res.Calls, res.Usage, res.Content)
		if res.Failure != nil {
			t.Logf("run %d failure: %+v", i+1, *res.Failure)
		}
		for _, p := range checkAcceptance(res) {
			t.Errorf("run %d: %s", i+1, p)
		}
		if first == nil {
			first = res
		} else if !reflect.DeepEqual(Canonical(res.Trace), Canonical(first.Trace)) {
			t.Errorf("run %d differs from run 1: the task isn't reproducible on this server", i+1)
		}
	}
}

// tinyURL returns the server the tiny-checkpoint tests talk to, or skips.
func tinyURL(t *testing.T) string {
	t.Helper()
	url := os.Getenv("KOLIBRI_BRIDGE_TINY_URL")
	if url == "" {
		t.Skip("KOLIBRI_BRIDGE_TINY_URL is not set: this needs llama-server on the cmd/kolibri-tiny checkpoint, see docs/execution-bridge.md")
	}
	return url
}

// forceCall makes a model of random weights emit a call: a final assistant message that opens the
// block, the non-lazy grammar of tool_choice "required" from there, and a stop string where the
// block closes. It proves nothing about what a model would choose. It does make the real server's
// parser, grammar and template handle everything the bridge sends and receives.
func forceCall(body map[string]json.RawMessage) {
	var messages []json.RawMessage
	if err := json.Unmarshal(body["messages"], &messages); err != nil {
		panic(err)
	}
	messages = append(messages, json.RawMessage(`{"role":"assistant","content":"<tool_call>\n"}`))
	encoded, err := json.Marshal(messages)
	if err != nil {
		panic(err)
	}
	body["messages"] = encoded
	body["tool_choice"] = json.RawMessage(`"required"`)
	body["stop"] = json.RawMessage(`["</tool_call>"]`)
	delete(body, "logit_bias")
}

const tinyEOSBody = `{"messages":[{"role":"user","content":"Add 2 and 3."}],"temperature":0,"seed":7,` +
	`"reasoning_effort":"none","max_tokens":1500,"logit_bias":[[127906,1000]]}`

// tinyLimits leave room for what a random model writes under a grammar: several calls in a turn.
func tinyLimits() Limits {
	lim := testLimits()
	lim.MaxCallsPerTurn = 32
	lim.MaxCalls = 128
	return lim
}

// tinyAllow keeps the prompt small: the tiny checkpoint's context is 2048 tokens, and every tool
// together takes 1931 of them. TestTinyCheckpointCompilesEverySchema covers the rest.
func tinyAllow() []string { return []string{"array_sum", "hash_sha256", "int_add", "int_div"} }

// Through the real llama-server on the tiny checkpoint: a forced call on the first turn, then a turn
// that hits EOS. The real parser splits the calls, and the real template renders the assistant and
// tool messages the bridge sends back.
func TestTinyCheckpointToolLoop(t *testing.T) {
	url := tinyURL(t)
	var first *Result
	for range 2 {
		b, err := New(Config{
			Upstream: url, Limits: tinyLimits(), OpLimits: ops.DefaultLimits(), Allow: tinyAllow(),
			RewriteTurn: func(turn int, body map[string]json.RawMessage) {
				if turn == 1 {
					forceCall(body)
				}
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		res, err := b.Run(context.Background(), "tiny", []byte(tinyEOSBody), Callbacks{})
		if err != nil {
			t.Fatal(err)
		}
		_ = b.Close(context.Background())
		wantStop(t, res, StopCompleted)
		if res.Turns != 2 || res.Calls < 1 {
			t.Fatalf("%d turns, %d calls", res.Turns, res.Calls)
		}
		t.Logf("usage %+v", res.Usage)
		for _, e := range res.Trace {
			switch e.Kind {
			case EventCallStart:
				t.Logf("the tiny model called %s %s %q", e.CallID, e.Op, e.Args)
			case EventCallEnd:
				t.Logf("result of %s, %s: %s", e.CallID, e.JobState, e.Result)
				var resp ops.Response
				if err := json.Unmarshal([]byte(e.Result), &resp); err != nil || resp.Version != ops.Version || resp.ID != e.CallID {
					t.Errorf("response %s: %v", e.Result, err)
				}
			}
		}
		if first == nil {
			first = res
		} else if !reflect.DeepEqual(Canonical(res.Trace), Canonical(first.Trace)) {
			t.Errorf("two runs differ:\n%+v\n%+v", Canonical(first.Trace), Canonical(res.Trace))
		}
	}
}

// The model that never stops calling, on the real server.
func TestTinyCheckpointTurnLimit(t *testing.T) {
	url := tinyURL(t)
	lim := tinyLimits()
	lim.MaxTurns = 3
	b, err := New(Config{
		Upstream: url, Limits: lim, OpLimits: ops.DefaultLimits(), Allow: tinyAllow(),
		RewriteTurn: func(_ int, body map[string]json.RawMessage) { forceCall(body) },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = b.Close(context.Background()) }()
	res, err := b.Run(context.Background(), "tiny-limit", []byte(tinyEOSBody), Callbacks{})
	if err != nil {
		t.Fatal(err)
	}
	wantStop(t, res, StopTurnLimit)
	if res.Turns != 3 || res.Calls < 2 {
		t.Fatalf("%d turns, %d calls", res.Turns, res.Calls)
	}
	for _, e := range res.Trace {
		if e.Kind == EventCallStart && e.Turn == 3 {
			t.Errorf("the last turn's call ran: %+v", e)
		}
	}
}

// The real server builds a grammar from every operation's schema, patterns included, and renders them
// all into the prompt. A schema it can't compile would be HTTP 400 on the first turn.
func TestTinyCheckpointCompilesEverySchema(t *testing.T) {
	url := tinyURL(t)
	f := newFakeFunc(t, func(int, []byte) fakeTurn { return fakeTurn{content: "x"} })
	b := newBridge(t, f)
	body, err := json.Marshal(map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "x"}},
		"tools":    b.Tools(), "tool_choice": "required", "max_tokens": 1, "verbose": true,
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(strings.TrimSuffix(url, "/")+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("%d: %s", resp.StatusCode, raw)
	}
	var out struct {
		Usage   wireUsage `json:"usage"`
		Verbose struct {
			Settings struct {
				Grammar string `json:"grammar"`
			} `json:"generation_settings"`
		} `json:"__verbose"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	for _, o := range b.Registry().Allowed() {
		if !strings.Contains(out.Verbose.Settings.Grammar, `"`+o.Name+`"`) {
			t.Errorf("%s isn't in the grammar", o.Name)
		}
	}
	// The integer pattern reached the grammar as the unbounded form the bridge depends on.
	if !strings.Contains(out.Verbose.Settings.Grammar, `[1-9] [0-9]*`) {
		t.Errorf("the grammar bounds integers:\n%s", out.Verbose.Settings.Grammar)
	}
	t.Logf("every tool: %d prompt tokens", out.Usage.PromptTokens)
}

// Arguments that aren't JSON would make the real server answer 500 on the next turn if they went
// back as they are. The first turn is scripted, the second goes to the real server.
func TestTinyCheckpointAcceptsTheEchoOfMalformedArguments(t *testing.T) {
	url := tinyURL(t)
	for _, c := range []struct {
		op, args string
		code     ops.Code
	}{
		{"int_add", `{`, ops.CodeMalformedArguments},
		{"int_add", `nonsense`, ops.CodeMalformedArguments},
		{"int_add", ``, ops.CodeMalformedArguments},
		{"int_add", `{"a":"1"`, ops.CodeMalformedArguments},
		// JSON to encoding/json, not to the server. The surrogate even runs: only the echo can fail it.
		{"hash_sha256", `{"data":"\ud800"}`, ""},
		{"hash_sha256", `{"data":"\udc00"}`, ""},
		{"int_add", `{"a":1e999,"b":"2"}`, ops.CodeInvalidArgument},
		{"int_add", `{"a":-1e999,"b":"2"}`, ops.CodeInvalidArgument},
		// JSON to both, but not an object.
		{"int_add", `5`, ops.CodeMalformedArguments},
		{"int_add", `null`, ops.CodeMalformedArguments},
		{"int_add", `"x"`, ops.CodeMalformedArguments},
		{"int_add", `[1]`, ops.CodeMalformedArguments},
	} {
		op, args := c.op, c.args
		f := newFakeFunc(t, func(turn int, body []byte) fakeTurn {
			if turn == 0 {
				return fakeTurn{calls: []fakeCall{{op, args}}}
			}
			resp, err := http.Post(strings.TrimSuffix(url, "/")+"/v1/chat/completions", "application/json", bytes.NewReader(body))
			if err != nil {
				t.Error(err)
				return fakeTurn{content: "no server"}
			}
			defer resp.Body.Close()
			raw, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Error(err)
			}
			return fakeTurn{raw: string(raw), rawCT: resp.Header.Get("Content-Type"), status: resp.StatusCode}
		})
		res := runBody(t, newBridge(t, f, func(cfg *Config) { cfg.Allow = tinyAllow() }), tinyEOSBody)
		wantStop(t, res, StopCompleted)
		if ends := callEnds(res); len(ends) != 1 || ends[0].Code != string(c.code) {
			t.Fatalf("%q: results %+v", args, ends)
		}
	}
}

// The control for the test above: the server does refuse those arguments unwrapped.
func TestTinyCheckpointRefusesRawMalformedArguments(t *testing.T) {
	url := tinyURL(t)
	body := `{"max_tokens":1,"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"","tool_calls":[{"id":"c","type":"function","function":{"name":"int_add","arguments":"{"}}]},{"role":"tool","tool_call_id":"c","content":"r"}]}`
	resp, err := http.Post(strings.TrimSuffix(url, "/")+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusInternalServerError || !strings.Contains(string(raw), "Failed to parse tool call arguments") {
		t.Fatalf("%d: %s", resp.StatusCode, raw)
	}
}

// The reason chat_template_kwargs is refused: its keys override the template's variables. The bridge
// delivers a result by appending to messages, so that this could keep it from the model.
func TestTinyCheckpointTemplateKwargsReplaceTheMessages(t *testing.T) {
	url := tinyURL(t)
	body := `{"messages":[{"role":"user","content":"real"}],"chat_template_kwargs":{"messages":[{"role":"user","content":"replaced"}]}}`
	resp, err := http.Post(strings.TrimSuffix(url, "/")+"/apply-template", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(raw), "replaced") || strings.Contains(string(raw), "real") {
		t.Fatalf("%d: %s", resp.StatusCode, raw)
	}
}

// The reason integers go as strings. The real server parses a call's arguments to render the next
// prompt, and its JSON library doesn't keep a number past int64.
func TestTinyCheckpointCorruptsLargeJSONNumbers(t *testing.T) {
	url := tinyURL(t)
	render := func(args string) string {
		t.Helper()
		body, err := json.Marshal(map[string]any{"messages": []any{
			map[string]any{"role": "user", "content": "x"},
			map[string]any{"role": "assistant", "content": "", "tool_calls": []any{map[string]any{
				"id": "c", "type": "function", "function": map[string]any{"name": "f", "arguments": args},
			}}},
			map[string]any{"role": "tool", "tool_call_id": "c", "content": "r"},
		}})
		if err != nil {
			t.Fatal(err)
		}
		resp, err := http.Post(strings.TrimSuffix(url, "/")+"/apply-template", "application/json", bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out struct {
			Prompt string `json:"prompt"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			t.Fatal(err)
		}
		return out.Prompt
	}
	for args, want := range map[string]string{
		`{"a":9223372036854775807}`:       `{"a": 9223372036854775807}`,       // int64's maximum survives
		`{"a":9223372036854775808}`:       `{"a": -9223372036854775808}`,      // 2^63 changes sign
		`{"a":99999999999999999999999}`:   `{"a": 1e+23}`,                     // rounded
		`{"a":"99999999999999999999999"}`: `{"a": "99999999999999999999999"}`, // a string is exact
	} {
		if got := render(args); !strings.Contains(got, want) {
			t.Errorf("%s: want %s in the prompt:\n%s", args, want, got)
		}
	}
}

// TestCaptureFixtures rewrites testdata/bridge from a running tiny checkpoint: the real bytes that
// the replay tests use. KOLIBRI_BRIDGE_CAPTURE_URL is a server with --jinja, -c 2048 and -np 1;
// KOLIBRI_BRIDGE_CAPTURE_NOJINJA_URL is the same with --no-jinja.
func TestCaptureFixtures(t *testing.T) {
	url := os.Getenv("KOLIBRI_BRIDGE_CAPTURE_URL")
	noJinja := os.Getenv("KOLIBRI_BRIDGE_CAPTURE_NOJINJA_URL")
	if url == "" || noJinja == "" {
		t.Skip("KOLIBRI_BRIDGE_CAPTURE_URL and KOLIBRI_BRIDGE_CAPTURE_NOJINJA_URL are not set")
	}
	const intProp = `{"type":"string","pattern":"^-?(0|[1-9][0-9]*)$"}`
	tools := json.RawMessage(`[{"type":"function","function":{"name":"int_add","description":"Add","parameters":{"type":"object","properties":{"a":` +
		intProp + `,"b":` + intProp + `},"required":["a","b"],"additionalProperties":false}}}]`)
	user := map[string]any{"role": "user", "content": "Add 2 and 3."}
	pre := map[string]any{"role": "assistant", "content": "<tool_call>\n"}
	base := func(extra map[string]any) map[string]any {
		out := map[string]any{"seed": 7, "temperature": 0, "reasoning_effort": "none", "tools": tools}
		for k, v := range extra {
			out[k] = v
		}
		return out
	}
	stream := func(extra map[string]any) map[string]any {
		out := base(extra)
		out["stream"] = true
		out["stream_options"] = map[string]any{"include_usage": true}
		return out
	}
	model := regexp.MustCompile(`"model":"[^"]*kolibri-tiny-f32\.gguf"`)
	capture := func(name, server string, body map[string]any) []byte {
		t.Helper()
		encoded, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := http.Post(strings.TrimSuffix(server, "/")+"/v1/chat/completions", "application/json", bytes.NewReader(encoded))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		raw, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		raw = model.ReplaceAll(raw, []byte(`"model":"kolibri-tiny-f32.gguf"`))
		if err := os.WriteFile(filepath.Join("..", "..", "testdata", "bridge", name), raw, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: HTTP %d, %d bytes", name, resp.StatusCode, len(raw))
		return raw
	}
	forced := map[string]any{"messages": []any{user, pre}, "tool_choice": "required", "stop": []string{"</tool_call>"}, "max_tokens": 200}
	cut := map[string]any{"messages": []any{user, pre}, "tool_choice": "required", "max_tokens": 80}
	garbage := map[string]any{"messages": []any{user, pre}, "tool_choice": "auto", "max_tokens": 80}
	eos := [][]int{{127906, 1000}}

	capture("tool_call.sse", url, stream(forced))
	whole := capture("tool_call.json", url, base(forced))
	capture("tool_call_cut.sse", url, stream(cut))
	capture("parse_error.sse", url, stream(garbage))
	capture("parse_error.json", url, base(garbage))
	capture("answer.sse", url, stream(map[string]any{"messages": []any{user}, "logit_bias": eos, "max_tokens": 20}))
	capture("answer_len.sse", url, stream(map[string]any{"messages": []any{user}, "max_tokens": 8}))
	capture("answer_len.json", url, base(map[string]any{"messages": []any{user}, "max_tokens": 8}))
	capture("reasoning.sse", url, map[string]any{
		"seed": 7, "temperature": 0, "tools": tools, "stream": true, "stream_options": map[string]any{"include_usage": true},
		"messages":               []any{user, map[string]any{"role": "assistant", "content": "", "reasoning_content": "Let me"}},
		"continue_final_message": "reasoning_content", "add_generation_prompt": false, "max_tokens": 6,
	})
	capture("exceed_context.json", url, stream(map[string]any{"messages": []any{map[string]any{"role": "user", "content": strings.Repeat("word ", 6000)}}, "max_tokens": 8}))
	capture("bad_request.json", url, map[string]any{"messages": "nope", "stream": true})
	capture("no_jinja.json", noJinja, stream(map[string]any{"messages": []any{user}, "max_tokens": 8}))

	var parsed struct {
		Choices []struct {
			Message struct {
				ToolCalls []struct {
					ID string `json:"id"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	var message map[string]any
	if err := json.Unmarshal(whole, &parsed); err != nil || len(parsed.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("%v: %s", err, whole)
	}
	var full struct {
		Choices []struct {
			Message map[string]any `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(whole, &full); err != nil {
		t.Fatal(err)
	}
	message = full.Choices[0].Message
	capture("second.json", url, base(map[string]any{
		"messages": []any{user, message, map[string]any{
			"role": "tool", "tool_call_id": parsed.Choices[0].Message.ToolCalls[0].ID,
			"content": `{"v":1,"id":"t1c0","op":"int_add","ok":true,"result":{"value":"144431509"}}`,
		}},
		"logit_bias": eos, "max_tokens": 20,
	}))
}
