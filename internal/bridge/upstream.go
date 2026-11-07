package bridge

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"mime"
	"net/http"
	"slices"
	"strings"
)

// StopReason says how a request ended. The values are part of the protocol.
type StopReason string

const (
	StopCompleted        StopReason = "completed"            // the model answered without a call
	StopTurnLimit        StopReason = "turn_limit"           // it asked for a call on its last allowed turn
	StopCallLimit        StopReason = "call_limit"           // it asked for more calls than allowed
	StopTruncated        StopReason = "truncated"            // a turn ended on finish_reason "length"
	StopUnparsedCall     StopReason = "unparsed_call"        // an answer has a tool-call marker that llama-server didn't parse
	StopStreamBroken     StopReason = "stream_broken"        // the stream ended before [DONE]
	StopUnreachable      StopReason = "upstream_unreachable" // no connection to the server
	StopUpstreamHTTP     StopReason = "upstream_http"        // the server refused the request with an HTTP error
	StopUpstreamError    StopReason = "upstream_error"       // the server sent an error event inside the stream
	StopBadUpstream      StopReason = "bad_upstream"         // the response isn't what llama-server sends
	StopResponseTooLarge StopReason = "response_too_large"   // a turn's response is past MaxTurnBytes
	StopTurnTimeout      StopReason = "turn_timeout"         // a turn ran past TurnTimeout
	StopRequestTimeout   StopReason = "request_timeout"      // the request ran past RequestTimeout
	StopCancelled        StopReason = "cancelled"            // the client or a cancel call ended it
	StopInternal         StopReason = "internal"             // a bug in the bridge or the scheduler
)

// Failure is why a request stopped early.
type Failure struct {
	Reason       StopReason `json:"reason"`
	Message      string     `json:"message"`
	HTTPStatus   int        `json:"http_status,omitempty"`   // the server's status, for upstream_http
	UpstreamType string     `json:"upstream_type,omitempty"` // the server's error type, such as exceed_context_size_error
}

func (f *Failure) Error() string { return string(f.Reason) + ": " + f.Message }

func failure(reason StopReason, format string, args ...any) *Failure {
	return &Failure{Reason: reason, Message: fmt.Sprintf(format, args...)}
}

// ToolCall is one complete call of a turn, as llama-server parsed it.
type ToolCall struct {
	UpstreamID string // random per response, kept for the trace and nowhere else
	Name       string
	Arguments  string // the model's text, byte for byte
}

// Usage is llama-server's token count of a turn or of the whole request.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	CachedTokens     int `json:"cached_tokens"`
}

// Turn is one completion.
type Turn struct {
	Model     string
	Reasoning string
	Content   string
	Calls     []ToolCall
	Finish    string
	Usage     Usage
}

// Delta is streamed text of a turn, for a client that streams.
type Delta struct {
	RequestID string
	Reasoning string
	Content   string
	Model     string
}

// splitLines is bufio.ScanLines without the last, unterminated line. Without a newline a line may
// be cut anywhere, and it must read as a stream that broke rather than as an event that isn't JSON.
func splitLines(data []byte, atEOF bool) (int, []byte, error) {
	if i := bytes.IndexByte(data, '\n'); i >= 0 {
		return i + 1, bytes.TrimSuffix(data[:i], []byte("\r")), nil
	}
	if atEOF {
		return len(data), nil, nil
	}
	return 0, nil, nil
}

// client talks to one server, and only to it: no proxy from the environment and no redirects, so a
// prompt can't end up on another host.
type client struct {
	url  string
	http *http.Client
}

func newClient(base string) *client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	return &client{
		url: strings.TrimSuffix(base, "/") + "/v1/chat/completions",
		http: &http.Client{
			Transport:     transport,
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		},
	}
}

// complete runs one turn. A nil *Failure means a response that ended properly, whatever its finish
// reason: the caller decides what a turn may do. With a failure, the turn may be nil or partial.
func (c *client) complete(ctx context.Context, body []byte, lim Limits, onDelta func(Delta)) (*Turn, *Failure) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(body))
	if err != nil {
		return nil, failure(StopInternal, "building the request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, failure(StopUnreachable, "the request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, statusFailure(resp, lim)
	}
	limited := &limitedBody{r: resp.Body, left: lim.MaxTurnBytes}
	media, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	switch media {
	case "text/event-stream":
		return readStream(limited, lim, onDelta)
	case "application/json":
		return readWhole(limited, lim, onDelta)
	}
	return nil, failure(StopBadUpstream, "unexpected content type %q", resp.Header.Get("Content-Type"))
}

// limitedBody ends a response past the limit with an error of its own, which io.LimitReader's plain
// EOF would hide: it would read as a stream that broke. A response of exactly the limit is not past
// it, so at the limit one more byte decides.
type limitedBody struct {
	r    io.Reader
	left int64
}

var errTooLarge = errors.New("response too large")

func (l *limitedBody) Read(p []byte) (int, error) {
	if l.left <= 0 {
		var probe [1]byte
		n, err := l.r.Read(probe[:])
		if n > 0 {
			return 0, errTooLarge
		}
		return 0, err
	}
	if int64(len(p)) > l.left {
		p = p[:l.left]
	}
	n, err := l.r.Read(p)
	l.left -= int64(n)
	return n, err
}

// upstreamError is llama-server's error object, format_error_response in server-common.cpp.
type upstreamError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Type    string `json:"type"`
}

func statusFailure(resp *http.Response, lim Limits) *Failure {
	const limit = 64 << 10
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, min(limit, lim.MaxTurnBytes)))
	f := failure(StopUpstreamHTTP, "HTTP %d", resp.StatusCode)
	f.HTTPStatus = resp.StatusCode
	var parsed struct {
		Error *upstreamError `json:"error"`
	}
	if json.Unmarshal(raw, &parsed) == nil && parsed.Error != nil {
		f.Message = parsed.Error.Message
		f.UpstreamType = parsed.Error.Type
	}
	return f
}

type callDelta struct {
	Index    *int   `json:"index"`
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type wireUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	Details          struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

func (u *wireUsage) usage() Usage {
	if u == nil {
		return Usage{}
	}
	return Usage{u.PromptTokens, u.CompletionTokens, u.TotalTokens, u.Details.CachedTokens}
}

// chunk is one server-sent event: to_json_oaicompat_chat and to_json_oaicompat_chat_stream in
// server-task.cpp, or an error.
type chunk struct {
	Error   *upstreamError `json:"error"`
	Model   string         `json:"model"`
	Usage   *wireUsage     `json:"usage"`
	Choices []struct {
		Index        int     `json:"index"`
		FinishReason *string `json:"finish_reason"`
		Delta        struct {
			Content          *string     `json:"content"`
			ReasoningContent *string     `json:"reasoning_content"`
			ToolCalls        []callDelta `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
}

// assembler builds the calls of a turn out of their fragments.
type assembler struct {
	lim   Limits
	calls map[int]*ToolCall
}

func (a *assembler) add(d callDelta) *Failure {
	index := len(a.calls) - 1
	if d.Index != nil {
		index = *d.Index
	}
	if index < 0 {
		return failure(StopBadUpstream, "tool-call fragment without an index")
	}
	call := a.calls[index]
	if call == nil {
		if len(a.calls) >= a.lim.MaxCallsPerTurn {
			return failure(StopCallLimit, "a turn has more than %d calls", a.lim.MaxCallsPerTurn)
		}
		call = &ToolCall{}
		a.calls[index] = call
	}
	if d.ID != "" {
		call.UpstreamID = d.ID
	}
	switch {
	case d.Function.Name == "":
	case call.Name == "":
		call.Name = d.Function.Name
	case call.Name != d.Function.Name:
		return failure(StopBadUpstream, "call %d changes its name from %q to %q", index, call.Name, d.Function.Name)
	}
	call.Arguments += d.Function.Arguments
	return nil
}

func (a *assembler) done() []ToolCall {
	var out []ToolCall
	for _, index := range slices.Sorted(maps.Keys(a.calls)) {
		out = append(out, *a.calls[index])
	}
	return out
}

// readStream reads events until [DONE]. Only [DONE] ends a turn: llama-server always sends it after
// the finish chunk, and it doesn't send it after an error. With a failure, the turn is what arrived
// before it.
func readStream(body io.Reader, lim Limits, onDelta func(Delta)) (*Turn, *Failure) {
	turn := &Turn{}
	calls := &assembler{lim: lim, calls: map[int]*ToolCall{}}
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64<<10), int(min(lim.MaxTurnBytes, 1<<30)))
	scanner.Split(splitLines)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 || line[0] == ':' || bytes.HasPrefix(line, []byte("event:")) {
			continue
		}
		data, ok := bytes.CutPrefix(line, []byte("data:"))
		if !ok {
			return turn, failure(StopBadUpstream, "unexpected line in the stream: %s", snippet(line))
		}
		data = bytes.TrimPrefix(data, []byte(" "))
		if string(data) == "[DONE]" {
			turn.Calls = calls.done()
			if turn.Finish == "" {
				return turn, failure(StopBadUpstream, "[DONE] without a finish_reason")
			}
			return turn, nil
		}
		var c chunk
		if err := json.Unmarshal(data, &c); err != nil {
			return turn, failure(StopBadUpstream, "an event is not JSON: %v: %s", err, snippet(data))
		}
		if c.Error != nil {
			f := failure(StopUpstreamError, "%s", c.Error.Message)
			f.HTTPStatus, f.UpstreamType = c.Error.Code, c.Error.Type
			return turn, f
		}
		if c.Model != "" {
			turn.Model = c.Model
		}
		if c.Usage != nil {
			turn.Usage = c.Usage.usage()
		}
		for _, choice := range c.Choices {
			if choice.Index != 0 {
				return turn, failure(StopBadUpstream, "choice %d: only one choice is expected", choice.Index)
			}
			if d := choice.Delta; d.Content != nil || d.ReasoningContent != nil {
				delta := Delta{Model: turn.Model}
				if d.Content != nil {
					turn.Content += *d.Content
					delta.Content = *d.Content
				}
				if d.ReasoningContent != nil {
					turn.Reasoning += *d.ReasoningContent
					delta.Reasoning = *d.ReasoningContent
				}
				if onDelta != nil && delta != (Delta{Model: turn.Model}) {
					onDelta(delta)
				}
			}
			for _, d := range choice.Delta.ToolCalls {
				if f := calls.add(d); f != nil {
					return turn, f
				}
			}
			if choice.FinishReason != nil {
				turn.Finish = *choice.FinishReason
			}
		}
	}
	err := scanner.Err()
	switch {
	case errors.Is(err, errTooLarge), errors.Is(err, bufio.ErrTooLong):
		return turn, failure(StopResponseTooLarge, "the response passes %d bytes", lim.MaxTurnBytes)
	case err != nil:
		return turn, failure(StopStreamBroken, "the stream broke with %d call fragments held back: %v", len(calls.calls), err)
	}
	return turn, failure(StopStreamBroken, "the stream ended before [DONE] with %d calls held back", len(calls.calls))
}

// readWhole reads a non-streaming response. The bridge always asks for a stream, a server that
// ignores that still gets a fair reading.
func readWhole(body io.Reader, lim Limits, onDelta func(Delta)) (*Turn, *Failure) {
	raw, err := io.ReadAll(body)
	switch {
	case errors.Is(err, errTooLarge):
		return nil, failure(StopResponseTooLarge, "the response passes %d bytes", lim.MaxTurnBytes)
	case err != nil:
		return nil, failure(StopStreamBroken, "the response broke: %v", err)
	}
	var parsed struct {
		Model   string     `json:"model"`
		Usage   *wireUsage `json:"usage"`
		Choices []struct {
			FinishReason *string `json:"finish_reason"`
			Message      struct {
				Content          *string `json:"content"`
				ReasoningContent string  `json:"reasoning_content"`
				ToolCalls        []struct {
					ID       string `json:"id"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, failure(StopBadUpstream, "the response is not JSON: %v: %s", err, snippet(raw))
	}
	if len(parsed.Choices) != 1 || parsed.Choices[0].FinishReason == nil {
		return nil, failure(StopBadUpstream, "expected one finished choice: %s", snippet(raw))
	}
	choice := parsed.Choices[0]
	turn := &Turn{Model: parsed.Model, Reasoning: choice.Message.ReasoningContent, Finish: *choice.FinishReason, Usage: parsed.Usage.usage()}
	if choice.Message.Content != nil {
		turn.Content = *choice.Message.Content
	}
	if len(choice.Message.ToolCalls) > lim.MaxCallsPerTurn {
		return nil, failure(StopCallLimit, "a turn has %d calls, the limit is %d", len(choice.Message.ToolCalls), lim.MaxCallsPerTurn)
	}
	for _, c := range choice.Message.ToolCalls {
		turn.Calls = append(turn.Calls, ToolCall{UpstreamID: c.ID, Name: c.Function.Name, Arguments: c.Function.Arguments})
	}
	if onDelta != nil && (turn.Content != "" || turn.Reasoning != "") {
		onDelta(Delta{Reasoning: turn.Reasoning, Content: turn.Content, Model: turn.Model})
	}
	return turn, nil
}

func snippet(b []byte) string {
	const limit = 200
	if len(b) > limit {
		return string(b[:limit]) + "..."
	}
	return string(b)
}
