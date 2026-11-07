package bridge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"kolibri-llm/internal/bridge/ops"
	"kolibri-llm/internal/coproc"
)

// Limits bounds one request and the bridge. A turn is one completion, a call one operation.
type Limits struct {
	MaxTurns        int           // completions per request; a request that still calls on the last one stops
	MaxCalls        int           // operations per request
	MaxCallsPerTurn int           // operations per completion; more stops the request
	MaxRequests     int           // open requests
	MaxRunningOps   int           // operations running at once
	MaxTurnTokens   int           // max_tokens of a completion
	MaxTurnBytes    int64         // bytes of a completion's response
	MaxRequestBytes int64         // bytes of a client's request
	OpTimeout       time.Duration // one operation's run time
	TurnTimeout     time.Duration // one completion, prompt processing included
	RequestTimeout  time.Duration // the whole request
	WriteTimeout    time.Duration // one write to an HTTP client: one that stops reading can't hold a request
	BodyTimeout     time.Duration // reading an HTTP client's request body: it doesn't count as open yet
}

// DefaultLimits returns the limits of a default bridge. The durations leave room for the real model
// on a slow tier: a real turn can spend minutes on its prompt before its first event.
func DefaultLimits() Limits {
	return Limits{
		MaxTurns:        16,
		MaxCalls:        64,
		MaxCallsPerTurn: 8,
		MaxRequests:     8,
		MaxRunningOps:   8,
		MaxTurnTokens:   4096,
		MaxTurnBytes:    8 << 20,
		MaxRequestBytes: 16 << 20,
		OpTimeout:       5 * time.Second,
		TurnTimeout:     15 * time.Minute,
		RequestTimeout:  time.Hour,
		WriteTimeout:    30 * time.Second,
		BodyTimeout:     30 * time.Second,
	}
}

func (l Limits) validate() error {
	if l.MaxTurns <= 0 || l.MaxCalls <= 0 || l.MaxCallsPerTurn <= 0 || l.MaxRequests <= 0 || l.MaxRunningOps <= 0 ||
		l.MaxTurnTokens <= 0 || l.MaxTurnBytes <= 0 || l.MaxRequestBytes <= 0 || l.OpTimeout <= 0 ||
		l.TurnTimeout <= 0 || l.RequestTimeout <= 0 || l.WriteTimeout <= 0 || l.BodyTimeout <= 0 {
		return fmt.Errorf("bridge: every limit must be positive: %+v", l)
	}
	return nil
}

// Config configures a Bridge.
type Config struct {
	Upstream string // llama-server's base URL
	Limits   Limits
	Ops      []ops.Op // nil means ops.Builtin
	Allow    []string // nil allows every operation, empty none
	OpLimits ops.Limits
	// CallMarker in a finished answer means the server failed to parse a call: llama-server hands
	// text it can't parse back as plain content. Empty means Kolibri's "<tool_call>".
	CallMarker string
	// Trace, if set, gets every event of every request, in order per request.
	Trace func(Event)
	// RewriteTurn, if set, changes a turn's request just before it is sent. Only the tiny-checkpoint
	// test uses it, to make a model of random weights emit a call at all.
	RewriteTurn func(turn int, body map[string]json.RawMessage)
}

// Bridge runs requests. It is safe for concurrent use.
type Bridge struct {
	cfg    Config
	client *client
	reg    *ops.Registry
	cp     *coprocessor
	sched  *coproc.Scheduler
	tools  json.RawMessage

	mu   sync.Mutex
	runs map[string]*run
	last uint64
}

// New starts a bridge. Close ends it.
func New(cfg Config) (*Bridge, error) {
	if cfg.Upstream == "" {
		return nil, errors.New("bridge: no upstream")
	}
	if err := cfg.Limits.validate(); err != nil {
		return nil, err
	}
	if cfg.Ops == nil {
		cfg.Ops = ops.Builtin()
	}
	if cfg.CallMarker == "" {
		cfg.CallMarker = "<tool_call>"
	}
	reg, err := ops.NewRegistry(cfg.Ops, cfg.Allow, cfg.OpLimits)
	if err != nil {
		return nil, err
	}
	tools, err := json.Marshal(reg.Tools())
	if err != nil {
		return nil, err
	}
	cp := &coprocessor{reg: reg}
	sc := coproc.DefaultConfig()
	sc.MaxRequests = cfg.Limits.MaxRequests
	sc.MaxJobsPerRequest = cfg.Limits.MaxCalls
	sc.MaxRunning = cfg.Limits.MaxRunningOps
	sc.MaxResultBytes = cfg.OpLimits.MaxResultBytes + 4<<10 // room for the envelope
	sc.DefaultTimeout = cfg.Limits.OpTimeout
	sc.MaxTimeout = cfg.Limits.OpTimeout
	// Every call is Execute, Wait and ReadResult, every turn a Continue. The turn and call limits
	// stop a request first; this is the scheduler's own backstop.
	sc.MaxActionsPerRequest = cfg.Limits.MaxTurns + 3*cfg.Limits.MaxCalls
	sched, err := coproc.NewScheduler(cp, sc)
	if err != nil {
		return nil, err
	}
	return &Bridge{cfg: cfg, client: newClient(cfg.Upstream), reg: reg, cp: cp, sched: sched, tools: tools, runs: map[string]*run{}}, nil
}

// Close cancels the open requests and returns once their operations have ended or ctx is done.
func (b *Bridge) Close(ctx context.Context) error {
	err := b.sched.Shutdown(ctx)
	done := make(chan struct{})
	go func() { b.cp.wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-ctx.Done():
		return errors.Join(err, ctx.Err())
	}
	return err
}

// Tools returns the OpenAI tools the bridge offers the model.
func (b *Bridge) Tools() json.RawMessage { return b.tools }

// Registry returns the bridge's operations.
func (b *Bridge) Registry() *ops.Registry { return b.reg }

// coprocessor runs a job on the registry. The job's arguments are the model's raw text, and the
// registry is the one gate: an unknown operation or malformed arguments end as an ERROR job like any
// other failure, so that every call of a turn gets a result in the order the model made them.
type coprocessor struct {
	reg *ops.Registry
	wg  sync.WaitGroup
}

// operationFailure carries an operation's failed response through the scheduler.
type operationFailure struct{ resp ops.Response }

func (e *operationFailure) Error() string { return e.resp.Error.Error() }

// Start implements coproc.Coprocessor.
func (c *coprocessor) Start(ctx context.Context, job coproc.JobSpec, publish func(coproc.CompletionEvent) error) error {
	if len(job.Args) != 1 {
		return fmt.Errorf("bridge: job %s has %d arguments", job.ID, len(job.Args))
	}
	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		resp := c.reg.Execute(ctx, ops.Request{ID: string(job.ID), Op: job.Program, Args: []byte(job.Args[0])})
		ev := coproc.CompletionEvent{RequestID: job.RequestID, JobID: job.ID, Generation: job.Generation, Status: coproc.JobDone}
		if resp.OK {
			ev.Result = resp.Encode()
		} else {
			ev.Status, ev.Error = coproc.JobError, &operationFailure{resp}
		}
		_ = publish(ev) // a stopped scheduler has settled the job already
	}()
	return nil
}

// RequestError is a request the bridge refused before running anything.
type RequestError struct {
	Status  int
	Type    string
	Message string
}

func (e *RequestError) Error() string { return e.Type + ": " + e.Message }

func invalid(format string, args ...any) *RequestError {
	return &RequestError{Status: http.StatusBadRequest, Type: "invalid_request_error", Message: fmt.Sprintf(format, args...)}
}

// accepted are the fields of a client's request. llama-server reads many more, and some of them take
// the bridge's place or change what a turn can produce: tools and grammar do, but so do
// chat_template_kwargs, whose keys override the template's variables, messages and tools included,
// and n_cmpl, the server's name for n. So this lists what is known to be harmless and refuses the
// rest, rather than listing what is known to be harmful.
var accepted = map[string]bool{
	"messages": true, "stream": true, "tool_choice": true, "n": true, "max_tokens": true, // checked below
	"model": true, "user": true, "seed": true, "stop": true, "logit_bias": true, "reasoning_effort": true, "parallel_tool_calls": true,
	"cache_prompt": true, "temperature": true, "top_p": true, "top_k": true, "min_p": true, "typical_p": true,
	"presence_penalty": true, "frequency_penalty": true, "repeat_penalty": true, "repeat_last_n": true,
}

var requestIDPattern = regexp.MustCompile(`^[A-Za-z0-9._-]{1,64}$`)

// chat is a client's request, parsed.
type chat struct {
	fields   map[string]json.RawMessage // what passes through to the server unchanged
	messages []json.RawMessage
	stream   bool
	oneCall  bool // parallel_tool_calls is false: a turn may hold one call
}

func (b *Bridge) parseChat(raw []byte) (*chat, *RequestError) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
		return nil, invalid("the body is not a JSON object")
	}
	for name := range fields {
		if !accepted[name] {
			return nil, invalid("field %q is not accepted: the bridge sets it, it could change what a turn can produce, or the bridge doesn't know it", clipText(name))
		}
	}
	c := &chat{fields: fields}
	if err := json.Unmarshal(fields["messages"], &c.messages); err != nil || len(c.messages) == 0 {
		return nil, invalid("messages must be a non-empty array")
	}
	for i, m := range c.messages {
		var parsed struct {
			Role string `json:"role"`
		}
		if err := json.Unmarshal(m, &parsed); err != nil || parsed.Role == "" {
			return nil, invalid("messages[%d] has no role", i)
		}
	}
	delete(fields, "messages")
	if raw, ok := fields["stream"]; ok {
		if err := json.Unmarshal(raw, &c.stream); err != nil {
			return nil, invalid("stream must be a boolean")
		}
		delete(fields, "stream")
	}
	if raw, ok := fields["tool_choice"]; ok {
		var choice string
		if err := json.Unmarshal(raw, &choice); err != nil || choice != "auto" {
			return nil, invalid(`tool_choice can only be "auto"`)
		}
	}
	if raw, ok := fields["parallel_tool_calls"]; ok {
		var parallel bool
		if err := json.Unmarshal(raw, &parallel); err != nil {
			return nil, invalid("parallel_tool_calls must be a boolean")
		}
		c.oneCall = !parallel
	}
	if raw, ok := fields["n"]; ok {
		var n int
		if err := json.Unmarshal(raw, &n); err != nil || n != 1 {
			return nil, invalid("n can only be 1")
		}
	}
	tokens := b.cfg.Limits.MaxTurnTokens
	if raw, ok := fields["max_tokens"]; ok {
		if err := json.Unmarshal(raw, &tokens); err != nil || tokens <= 0 || tokens > b.cfg.Limits.MaxTurnTokens {
			return nil, invalid("max_tokens must be in 1..%d", b.cfg.Limits.MaxTurnTokens)
		}
	}
	fields["max_tokens"] = json.RawMessage(fmt.Sprint(tokens))
	return c, nil
}

// Callbacks gets what happens during a request, on the goroutine that runs it. A callback that blocks
// holds up that request, but not a cancel call.
type Callbacks struct {
	Delta func(Delta)
	Event func(Event)
}

// Result is the outcome of a request. A failed request still has its trace, its messages and the
// last turn's text, as far as that turn got. A turn that failed has no finish reason. The trace is
// complete: it ends in request_end. The messages are the client's, then each turn that went on with
// its results, then the answer, if the last turn gave one. A turn that ended the request any other
// way isn't among them: its text is in Content.
type Result struct {
	RequestID string            `json:"request_id"`
	Protocol  int               `json:"protocol"`
	Stop      StopReason        `json:"stop"`
	Failure   *Failure          `json:"failure,omitempty"`
	Content   string            `json:"content"`
	Reasoning string            `json:"reasoning,omitempty"`
	Finish    string            `json:"finish_reason,omitempty"`
	Model     string            `json:"model,omitempty"`
	Turns     int               `json:"turns"`
	Calls     int               `json:"calls"`
	Usage     Usage             `json:"usage"`
	Messages  []json.RawMessage `json:"messages"`
	Trace     []Event           `json:"trace"`
}

// Run executes one request to its end. id may be empty. The error is a *RequestError: nothing ran.
// Every other outcome, every kind of failure included, is a Result.
func (b *Bridge) Run(ctx context.Context, id string, body []byte, cb Callbacks) (*Result, error) {
	c, reqErr := b.parseChat(body)
	if reqErr != nil {
		return nil, reqErr
	}
	return b.start(ctx, id, c, cb)
}

func (b *Bridge) start(ctx context.Context, id string, c *chat, cb Callbacks) (*Result, error) {
	generated := id == ""
	if !generated && !requestIDPattern.MatchString(id) {
		return nil, invalid("request ID %q doesn't match %s", clipText(id), requestIDPattern)
	}
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	ctx, stopTimeout := context.WithTimeoutCause(ctx, b.cfg.Limits.RequestTimeout, errRequestTimeout)
	defer stopTimeout()

	r := &run{
		b: b, id: id, cancel: cancel, cb: cb, fields: c.fields, oneCall: c.oneCall, messages: c.messages, began: time.Now(),
		calls: map[coproc.JobID]*call{},
	}
	var req *coproc.Request
	var err error
	for {
		if generated {
			b.mu.Lock()
			b.last++
			id = fmt.Sprintf("req-%d", b.last)
			b.mu.Unlock()
			r.id = id
		}
		req, err = b.sched.Open(coproc.RequestID(id), r)
		// A client may have named its request req-7. One that named nothing can't be a duplicate.
		if !generated || !errors.Is(err, coproc.ErrDuplicateRequest) {
			break
		}
	}
	switch {
	case errors.Is(err, coproc.ErrDuplicateRequest):
		return nil, &RequestError{Status: http.StatusConflict, Type: "duplicate_request", Message: fmt.Sprintf("request %s is running", id)}
	case errors.Is(err, coproc.ErrTooManyRequests):
		return nil, &RequestError{Status: http.StatusTooManyRequests, Type: "too_many_requests", Message: "too many open requests"}
	case err != nil:
		return nil, &RequestError{Status: http.StatusServiceUnavailable, Type: "unavailable", Message: err.Error()}
	}
	defer req.Close()
	r.req = req

	b.mu.Lock()
	b.runs[id] = r
	b.mu.Unlock()
	defer func() {
		b.mu.Lock()
		delete(b.runs, id)
		b.mu.Unlock()
	}()

	var names []string
	for _, o := range b.reg.Allowed() {
		names = append(names, o.Name)
	}
	r.emit(Event{Kind: EventRequestStart, Protocol: ops.Version, Allowed: names})
	runErr := req.Run(ctx, &controller{r: r}, nil)
	r.finish(ctx, runErr)
	return r.result(), nil
}

var (
	errRequestTimeout = errors.New("the request ran past its time limit")
	errTurnTimeout    = errors.New("the turn ran past its time limit")
	errCancelled      = errors.New("cancelled by a cancel call")
)

// Cancel ends a running request. It reports whether it did: true means the request's stop reason is
// cancelled. A request whose request_end is recorded has its outcome, so it counts as not running,
// even while its client is still taking that event. Cancel doesn't wait for the client: see record.
func (b *Bridge) Cancel(id string) bool {
	b.mu.Lock()
	r := b.runs[id]
	b.mu.Unlock()
	if r == nil || !r.record(Event{Kind: EventCancelRequested}) {
		return false
	}
	r.cancel(errCancelled)
	return true
}

// CancelCall cancels one operation of a running request. The model gets the error "cancelled" as
// that call's result and the request goes on. An operation that already has a result keeps it, and
// the answer is false. As for Cancel, a request whose request_end is recorded isn't running.
func (b *Bridge) CancelCall(id, callID string) (bool, error) {
	b.mu.Lock()
	r := b.runs[id]
	b.mu.Unlock()
	if r == nil || !r.record(Event{Kind: EventCancelRequested, CallID: callID}) {
		return false, fmt.Errorf("no running request %q", id)
	}
	if err := r.req.Cancel(coproc.JobID(callID)); err != nil {
		return false, err
	}
	info, ok := r.req.Job(coproc.JobID(callID))
	return ok && info.State == coproc.JobCancelled, nil
}

// call is one tool call in flight.
type call struct {
	id   string
	name string
}

// run is one request. It is the coproc.Generator: a Step is a whole turn, the boundary the scheduler
// resumes at, and a Deliver appends a tool message.
type run struct {
	b       *Bridge
	id      string
	req     *coproc.Request
	cancel  context.CancelCauseFunc
	cb      Callbacks
	fields  map[string]json.RawMessage
	oneCall bool
	began   time.Time

	// Used by the goroutine in Run, which also runs Step and Deliver. The scheduler never calls
	// those two at once.
	messages []json.RawMessage
	calls    map[coproc.JobID]*call
	turns    int
	nCalls   int
	last     *Turn
	usage    Usage
	stop     StopReason
	fail     *Failure

	traceMu sync.Mutex
	seq     int
	trace   []Event
	sent    int  // how much of the trace has gone to the sink and the callback
	ended   bool // request_end has been recorded: a late cancel call finds nothing to report to
	// A cancel of the whole request has been recorded, and answered true. finish holds to that.
	cancelRequested bool
}

// record adds an event to the trace without telling anyone. A cancel call stops here, on its own
// goroutine: whatever the client is doing, it mustn't keep a cancel from taking effect or from being
// answered. The run's goroutine sends the event on at its next emit, and request_end always comes.
// It reports false once request_end is recorded, when the event is dropped.
func (r *run) record(e Event) bool {
	r.traceMu.Lock()
	defer r.traceMu.Unlock()
	return r.recordLocked(e)
}

func (r *run) recordLocked(e Event) bool {
	if r.ended {
		return false
	}
	r.ended = e.Kind == EventRequestEnd
	r.cancelRequested = r.cancelRequested || (e.Kind == EventCancelRequested && e.CallID == "")
	r.seq++
	e.Seq = r.seq
	e.RequestID = r.id
	e.ElapsedMS = time.Since(r.began).Milliseconds()
	r.trace = append(r.trace, e)
	return true
}

// emit records an event and sends what the trace holds so far on.
func (r *run) emit(e Event) {
	r.record(e)
	r.send()
}

// send passes on what the trace holds and hasn't passed on yet. Only the run's goroutine calls it, so
// the events go out in order, and no lock is held while a slow client takes them.
func (r *run) send() {
	r.traceMu.Lock()
	pending := r.trace[r.sent:]
	r.sent = len(r.trace)
	r.traceMu.Unlock()
	for _, e := range pending {
		if r.b.cfg.Trace != nil {
			r.b.cfg.Trace(e)
		}
		if r.cb.Event != nil {
			r.cb.Event(e)
		}
	}
}

// Step implements coproc.Generator: one completion.
func (r *run) Step(ctx context.Context) (string, error) {
	r.turns++
	body, err := r.requestBody()
	if err != nil {
		return "", failure(StopInternal, "encoding the request: %v", err)
	}
	r.emit(Event{Kind: EventTurnStart, Turn: r.turns, BodySHA256: sha256Hex(body)})

	turnCtx, stop := context.WithTimeoutCause(ctx, r.b.cfg.Limits.TurnTimeout, errTurnTimeout)
	defer stop()
	onDelta := r.cb.Delta
	if onDelta != nil {
		onDelta = func(d Delta) { d.RequestID = r.id; r.cb.Delta(d) }
	}
	// The server may not hold to parallel_tool_calls: under tool_choice "required" it didn't.
	lim := r.b.cfg.Limits
	if r.oneCall {
		lim.MaxCallsPerTurn = 1
	}
	turn, f := r.b.client.complete(turnCtx, body, lim, onDelta)
	if f != nil {
		// A cut connection reads as a broken stream. Say what really cut it.
		if cause := r.causeFailure(context.Cause(turnCtx)); cause != nil {
			f = cause
		}
		r.last = r.failedTurn(turn)
		r.emit(Event{Kind: EventTurnEnd, Turn: r.turns, Code: string(f.Reason)})
		return "", f
	}
	r.last = turn
	r.usage.PromptTokens += turn.Usage.PromptTokens
	r.usage.CompletionTokens += turn.Usage.CompletionTokens
	r.usage.TotalTokens += turn.Usage.TotalTokens
	r.usage.CachedTokens += turn.Usage.CachedTokens
	usage := turn.Usage
	r.emit(Event{
		Kind: EventTurnEnd, Turn: r.turns, Finish: turn.Finish, Calls: len(turn.Calls),
		Content: clipText(turn.Content), Reasoning: clipText(turn.Reasoning), Usage: &usage,
	})
	return turn.Content, nil
}

// failedTurn is what's left of a turn that failed, which may be nothing: its text so far and no
// finish reason, even if the finish chunk arrived. It is the last turn now, not the one before it.
func (r *run) failedTurn(turn *Turn) *Turn {
	failed := &Turn{}
	if turn != nil {
		failed.Content, failed.Reasoning, failed.Model = turn.Content, turn.Reasoning, turn.Model
	}
	if failed.Model == "" && r.last != nil {
		failed.Model = r.last.Model
	}
	return failed
}

// requestBody builds a turn's request. encoding/json sorts the keys of a map, so that the same
// messages give the same bytes.
func (r *run) requestBody() ([]byte, error) {
	body := maps.Clone(r.fields)
	messages, err := json.Marshal(r.messages)
	if err != nil {
		return nil, err
	}
	body["messages"] = messages
	body["tools"] = r.b.tools
	body["stream"] = json.RawMessage("true")
	body["stream_options"] = json.RawMessage(`{"include_usage":true}`)
	if r.b.cfg.RewriteTurn != nil {
		r.b.cfg.RewriteTurn(r.turns, body)
	}
	return json.Marshal(body)
}

// Deliver implements coproc.Generator: one result becomes one tool message.
func (r *run) Deliver(_ context.Context, msg coproc.ResultMessage) error {
	c := r.calls[msg.Job]
	if c == nil {
		return fmt.Errorf("bridge: result of unknown call %s", msg.Job)
	}
	var content, code string
	var opFail *operationFailure
	switch {
	case msg.Status == coproc.JobDone:
		content = msg.Result
	case errors.As(msg.Err, &opFail):
		content, code = opFail.resp.Encode(), string(opFail.resp.Error.Code)
	default:
		resp := r.schedulerFailure(c, msg)
		content, code = resp.Encode(), string(resp.Error.Code)
	}
	encoded, err := json.Marshal(toolMessage{Role: "tool", ToolCallID: c.id, Content: content})
	if err != nil {
		return err
	}
	r.messages = append(r.messages, encoded)
	r.emit(Event{
		Kind: EventCallEnd, Turn: r.turns, CallID: c.id, Op: c.name, JobState: msg.Status.String(), Code: code,
		Result: clipText(content), ResultSHA256: sha256Hex([]byte(content)),
	})
	return nil
}

// schedulerFailure is the response of a job the scheduler ended rather than the operation.
func (r *run) schedulerFailure(c *call, msg coproc.ResultMessage) ops.Response {
	switch {
	case msg.Status == coproc.JobTimeout:
		return r.b.reg.Failure(c.id, c.name, ops.CodeTimeout, fmt.Sprintf("the operation ran past %v", r.b.cfg.Limits.OpTimeout))
	case msg.Status == coproc.JobCancelled:
		return r.b.reg.Failure(c.id, c.name, ops.CodeCancelled, "the operation was cancelled")
	case errors.Is(msg.Err, coproc.ErrResultTooLarge):
		return r.b.reg.Failure(c.id, c.name, ops.CodeResultTooLarge, "the result is past the scheduler's limit")
	}
	return r.b.reg.Failure(c.id, c.name, ops.CodeInternal, fmt.Sprintf("job %s: %v", msg.Status, msg.Err))
}

type toolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type toolCallMessage struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
}

type assistantMessage struct {
	Role             string            `json:"role"`
	Content          string            `json:"content"`
	ReasoningContent string            `json:"reasoning_content,omitempty"`
	ToolCalls        []toolCallMessage `json:"tool_calls,omitempty"`
}

type toolMessage struct {
	Role       string `json:"role"`
	ToolCallID string `json:"tool_call_id"`
	Content    string `json:"content"`
}

// echoArguments returns a call's arguments for the assistant message of the next prompt.
// llama-server parses them to render the template and answers HTTP 500 if it can't, so that arguments
// the model got wrong, whose error it should get to read, would end the request instead. JSON the
// server can parse goes back byte for byte. Anything else is wrapped, so that the model still sees
// what it wrote. The operation itself may have run: encoding/json takes some of what the server
// doesn't.
func echoArguments(args string) string {
	if json.Valid([]byte(args)) && serverCanParse(args) {
		return args
	}
	const limit = 256
	if len(args) > limit {
		args = strings.ToValidUTF8(args[:limit], "") + "..."
	}
	wrapped, err := json.Marshal(map[string]string{"_malformed": args})
	if err != nil {
		return "{}"
	}
	return string(wrapped)
}

// serverCanParse says whether llama-server's JSON parser, nlohmann's, takes text that json.Valid has
// taken. It is stricter there in three ways: invalid UTF-8, an escape of a lone surrogate, and a number
// that overflows a double, whether written as 1e999 or with 400 digits.
func serverCanParse(text string) bool {
	if !utf8.ValidString(text) {
		return false
	}
	for i := 0; i < len(text); {
		switch c := text[i]; {
		case c == '"':
			end, ok := scanString(text, i+1)
			if !ok {
				return false
			}
			i = end
		case c == '-' || (c >= '0' && c <= '9'):
			end := i + 1
			for end < len(text) && strings.IndexByte("+-.eE0123456789", text[end]) >= 0 {
				end++
			}
			if f, _ := strconv.ParseFloat(text[i:end], 64); math.IsInf(f, 0) {
				return false
			}
			i = end
		default:
			i++
		}
	}
	return true
}

// scanString reads a valid JSON string from just after its opening quote to just after its closing
// one. It reports false on an escape of a surrogate that isn't half of a pair.
func scanString(text string, i int) (end int, ok bool) {
	escapeAt := func(i int) (rune, bool) {
		if i+6 > len(text) || text[i] != '\\' || text[i+1] != 'u' {
			return 0, false
		}
		v, err := strconv.ParseUint(text[i+2:i+6], 16, 16)
		return rune(v), err == nil
	}
	for i < len(text) {
		switch text[i] {
		case '"':
			return i + 1, true
		case '\\':
			if v, isUnicode := escapeAt(i); isUnicode {
				if v >= 0xDC00 && v <= 0xDFFF {
					return 0, false
				}
				if v >= 0xD800 && v <= 0xDBFF {
					low, isEscape := escapeAt(i + 6)
					if !isEscape || low < 0xDC00 || low > 0xDFFF {
						return 0, false
					}
					i += 6
				}
				i += 6
				continue
			}
			i += 2
		default:
			i++
		}
	}
	return 0, false
}

// controller is the deterministic plan of a request: a turn, then for each of its calls Execute, and
// for each in order Wait and ReadResult, then the next turn. Results reach the model in the order of
// its calls, whichever finishes first.
type controller struct {
	r     *run
	queue []coproc.Action
}

// Next implements coproc.Controller. It stops on the first thing a request can't recover from.
func (c *controller) Next(obs coproc.Observation) (coproc.Action, bool) {
	r := c.r
	switch obs.Action.(type) {
	case nil: // the first call
		return coproc.Continue{}, true
	case coproc.Continue:
		if obs.Err != nil {
			r.failWith(obs.Err)
			return nil, false
		}
		if !c.planTurn() {
			return nil, false
		}
	default:
		if obs.Err != nil {
			r.failWith(obs.Err)
			return nil, false
		}
	}
	if len(c.queue) == 0 {
		return coproc.Continue{}, true
	}
	next := c.queue[0]
	c.queue = c.queue[1:]
	return next, true
}

// planTurn decides what a finished turn means. It reports whether the request goes on.
func (c *controller) planTurn() bool {
	r := c.r
	turn := r.last
	lim := r.b.cfg.Limits
	var stop *Failure
	switch {
	case turn.Finish == "length":
		// A turn out of tokens may hold a call that only looks complete, as in the captured
		// testdata/bridge/tool_call_cut.sse, or one with its closing tag missing. Nothing runs.
		stop = failure(StopTruncated, "turn %d ended on finish_reason length with %d calls", r.turns, len(turn.Calls))
	case turn.Finish == "stop" && len(turn.Calls) == 0:
		if strings.Contains(turn.Content, r.b.cfg.CallMarker) {
			stop = failure(StopUnparsedCall, "turn %d answered with %q in its content: the server did not parse it as a call", r.turns, r.b.cfg.CallMarker)
			break
		}
		// The answer is the end of the conversation. No prompt reads it, so it changes no request.
		encoded, err := json.Marshal(assistantMessage{Role: "assistant", Content: turn.Content, ReasoningContent: turn.Reasoning})
		if err != nil {
			stop = failure(StopInternal, "encoding the assistant message: %v", err)
			break
		}
		r.messages = append(r.messages, encoded)
		r.stop = StopCompleted
		return false
	case turn.Finish != "tool_calls" || len(turn.Calls) == 0:
		stop = failure(StopBadUpstream, "turn %d: finish_reason %q with %d calls", r.turns, turn.Finish, len(turn.Calls))
	case r.nCalls+len(turn.Calls) > lim.MaxCalls:
		stop = failure(StopCallLimit, "%d calls so far and %d more asked for, the limit is %d", r.nCalls, len(turn.Calls), lim.MaxCalls)
	case r.turns >= lim.MaxTurns:
		// A result would need another turn to be read. Nothing runs.
		stop = failure(StopTurnLimit, "turn %d of %d still asks for %d calls", r.turns, lim.MaxTurns, len(turn.Calls))
	}
	if stop != nil {
		for _, tc := range turn.Calls {
			r.emit(Event{
				Kind: EventCallRefused, Turn: r.turns, Op: tc.Name, UpstreamID: tc.UpstreamID, Code: string(stop.Reason),
				Args: clipText(tc.Arguments), ArgsSHA256: sha256Hex([]byte(tc.Arguments)),
			})
		}
		r.stopWith(stop)
		return false
	}

	msg := assistantMessage{Role: "assistant", Content: turn.Content, ReasoningContent: turn.Reasoning}
	var waits []coproc.Action
	for i, tc := range turn.Calls {
		// IDs are the bridge's, not the server's random ones, so that a run can be reproduced.
		id := fmt.Sprintf("t%dc%d", r.turns, i)
		job := coproc.JobID(id)
		r.calls[job] = &call{id: id, name: tc.Name}
		r.nCalls++
		msg.ToolCalls = append(msg.ToolCalls, toolCallMessage{
			ID: id, Type: "function",
			Function: toolFunction{Name: tc.Name, Arguments: echoArguments(tc.Arguments)},
		})
		r.emit(Event{
			Kind: EventCallStart, Turn: r.turns, CallID: id, Op: tc.Name, UpstreamID: tc.UpstreamID,
			Args: clipText(tc.Arguments), ArgsSHA256: sha256Hex([]byte(tc.Arguments)),
		})
		c.queue = append(c.queue, coproc.Execute{Job: job, Program: programName(tc.Name), Args: []string{tc.Arguments}})
		waits = append(waits, coproc.Wait{Job: job}, coproc.ReadResult{Job: job})
	}
	encoded, err := json.Marshal(msg)
	if err != nil {
		r.stopWith(failure(StopInternal, "encoding the assistant message: %v", err))
		return false
	}
	r.messages = append(r.messages, encoded)
	c.queue = append(c.queue, waits...)
	return true
}

// programName gives the scheduler a non-empty program name for a call without one. The registry
// answers it with unknown_op like any other name.
func programName(name string) string {
	if name == "" {
		return "(no name)"
	}
	return name
}

func (r *run) stopWith(f *Failure) {
	r.stop, r.fail = f.Reason, f
}

// failWith ends the request on an action's error.
func (r *run) failWith(err error) {
	var f *Failure
	switch {
	case errors.As(err, &f):
		r.stopWith(f)
	default:
		r.stopWith(failure(StopInternal, "scheduler: %v", err))
	}
}

// causeFailure says why a context ended, nil if it hasn't.
func (r *run) causeFailure(cause error) *Failure {
	lim := r.b.cfg.Limits
	switch {
	case cause == nil:
		return nil
	case errors.Is(cause, errTurnTimeout):
		return failure(StopTurnTimeout, "turn %d ran past %v", r.turns, lim.TurnTimeout)
	case errors.Is(cause, errRequestTimeout):
		return failure(StopRequestTimeout, "the request ran past %v", lim.RequestTimeout)
	case errors.Is(cause, context.DeadlineExceeded):
		return failure(StopRequestTimeout, "the caller's deadline passed")
	case errors.Is(cause, context.Canceled):
		// net/http cancels a request's context when a write to its client fails, before the write
		// returns to us, so that this covers a client that stopped reading as well.
		return failure(StopCancelled, "the client went away, or the bridge is closing")
	}
	return failure(StopCancelled, "%v", cause)
}

// finish decides the request's outcome once its controller has ended, and closes the trace.
func (r *run) finish(ctx context.Context, runErr error) {
	if r.stop == "" {
		f := r.causeFailure(context.Cause(ctx))
		switch {
		case f != nil:
		case errors.Is(runErr, coproc.ErrRequestClosed):
			f = failure(StopCancelled, "the bridge is shutting down")
		case runErr != nil:
			f = failure(StopInternal, "scheduler: %v", runErr)
		default:
			f = failure(StopInternal, "the controller ended without an outcome")
		}
		r.stopWith(f)
	}
	usage := r.usage
	// A cancel call answers true if it gets in before request_end. Taking the outcome and recording
	// it under one hold of the lock makes that answer hold, also for a cancel that came after the
	// outcome was decided: a turn that ran out of tokens, say, with a client holding up call_refused.
	r.traceMu.Lock()
	if r.cancelRequested && r.stop != StopCancelled {
		r.stopWith(failure(StopCancelled, "%v", errCancelled))
		r.last = r.failedTurn(r.last) // a failure has no finish reason
	}
	r.recordLocked(Event{Kind: EventRequestEnd, Stop: r.stop, Failure: r.fail, Turn: r.turns, Calls: r.nCalls, Usage: &usage})
	r.traceMu.Unlock()
	r.send()
}

func (r *run) result() *Result {
	r.traceMu.Lock()
	defer r.traceMu.Unlock()
	res := &Result{
		RequestID: r.id, Protocol: ops.Version, Stop: r.stop, Failure: r.fail,
		Turns: r.turns, Calls: r.nCalls, Usage: r.usage, Messages: r.messages, Trace: r.trace,
	}
	if r.last != nil {
		res.Content, res.Reasoning, res.Finish, res.Model = r.last.Content, r.last.Reasoning, r.last.Finish, r.last.Model
	}
	return res
}
