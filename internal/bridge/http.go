package bridge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"kolibri-llm/internal/bridge/ops"
)

// Handler returns the bridge's HTTP API.
//
//	POST /v1/chat/completions          OpenAI's request, without tools: the bridge brings its own
//	POST /v1/executions/{id}/cancel    ends a request; with {"call_id":"t1c0"} only that operation
//	GET  /v1/bridge/tools              the tools with their output schemas and the limits
//	GET  /health
func (b *Bridge) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", b.handleChat)
	mux.HandleFunc("POST /v1/executions/{id}/cancel", b.handleCancel)
	mux.HandleFunc("GET /v1/bridge/tools", b.handleTools)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"status":"ok"}`)
	})
	return mux
}

// extendWriteDeadline bounds the next write to a client of a chat request, whose slot or handler a
// client that stops reading could hold. The last one stays set for what net/http still flushes when
// the handler has returned, and net/http clears it after that, for the connection's next request. A
// writer without deadlines isn't a network connection, so nothing can get stuck in it.
func (b *Bridge) extendWriteDeadline(w http.ResponseWriter) {
	_ = http.NewResponseController(w).SetWriteDeadline(time.Now().Add(b.cfg.Limits.WriteTimeout))
}

// statusOf maps a stop reason to the status of a non-streaming response.
func statusOf(reason StopReason) int {
	switch reason {
	case StopCompleted:
		return http.StatusOK
	case StopTurnLimit, StopCallLimit, StopTruncated, StopUnparsedCall:
		return http.StatusUnprocessableEntity
	case StopTurnTimeout, StopRequestTimeout:
		return http.StatusGatewayTimeout
	case StopCancelled:
		return 499 // nginx's "client closed request": no standard status says this
	case StopInternal:
		return http.StatusInternalServerError
	}
	return http.StatusBadGateway
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// errorBody has the shape of llama-server's error object, and a stop reason in its type.
func errorBody(status int, kind, message string) map[string]any {
	return map[string]any{"error": map[string]any{"code": status, "type": kind, "message": message}}
}

// readBody reads a request's body within BodyTimeout, which a client that drips it would otherwise
// not be bound by: it holds a connection before it counts toward MaxRequests. The deadline is only
// for this read. A server-wide ReadTimeout would go on running during a long turn, and end its
// request's context through net/http's background read.
func (b *Bridge) readBody(w http.ResponseWriter, r *http.Request, limit int64) ([]byte, error) {
	rc := http.NewResponseController(w)
	_ = rc.SetReadDeadline(time.Now().Add(b.cfg.Limits.BodyTimeout))
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, limit))
	// An expired deadline stays: after the handler net/http reads what is left of a body, and the
	// client that dripped it would hold the connection there.
	if !errors.Is(err, os.ErrDeadlineExceeded) {
		_ = rc.SetReadDeadline(time.Time{})
	}
	return body, err
}

// bodyTooSlow answers a client whose body didn't arrive in time, and reports whether it was one.
func (b *Bridge) bodyTooSlow(w http.ResponseWriter, err error) bool {
	if !errors.Is(err, os.ErrDeadlineExceeded) {
		return false
	}
	w.Header().Set("Connection", "close")
	writeJSON(w, http.StatusRequestTimeout, errorBody(http.StatusRequestTimeout, "request_timeout",
		fmt.Sprintf("the body didn't arrive within %v", b.cfg.Limits.BodyTimeout)))
	return true
}

func (b *Bridge) handleChat(w http.ResponseWriter, r *http.Request) {
	body, err := b.readBody(w, r, b.cfg.Limits.MaxRequestBytes)
	b.extendWriteDeadline(w)
	if err != nil {
		if b.bodyTooSlow(w, err) {
			return
		}
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeJSON(w, http.StatusRequestEntityTooLarge, errorBody(http.StatusRequestEntityTooLarge, "request_too_large",
				fmt.Sprintf("the body is past %d bytes", b.cfg.Limits.MaxRequestBytes)))
			return
		}
		writeJSON(w, http.StatusBadRequest, errorBody(http.StatusBadRequest, "invalid_request_error", err.Error()))
		return
	}
	c, reqErr := b.parseChat(body)
	if reqErr != nil {
		writeJSON(w, reqErr.Status, errorBody(reqErr.Status, reqErr.Type, reqErr.Message))
		return
	}
	id := r.Header.Get("X-Request-Id")
	if c.stream {
		b.serveStream(w, r, id, c)
		return
	}
	res, err := b.start(r.Context(), id, c, Callbacks{})
	b.extendWriteDeadline(w)
	if err != nil {
		var refused *RequestError
		if errors.As(err, &refused) {
			writeJSON(w, refused.Status, errorBody(refused.Status, refused.Type, refused.Message))
			return
		}
		writeJSON(w, http.StatusInternalServerError, errorBody(http.StatusInternalServerError, "internal", err.Error()))
		return
	}
	if res.Stop != StopCompleted {
		out := errorBody(statusOf(res.Stop), string(res.Stop), res.Failure.Message)
		out["kolibri_bridge"] = res
		writeJSON(w, statusOf(res.Stop), out)
		return
	}
	writeJSON(w, http.StatusOK, completion(res))
}

// completion is OpenAI's chat.completion of a finished request, and the bridge's account of it.
func completion(res *Result) map[string]any {
	message := map[string]any{"role": "assistant", "content": res.Content}
	if res.Reasoning != "" {
		message["reasoning_content"] = res.Reasoning
	}
	return map[string]any{
		"id": res.RequestID, "object": "chat.completion", "created": time.Now().Unix(), "model": res.Model,
		"choices": []map[string]any{{"index": 0, "finish_reason": res.Finish, "message": message}},
		"usage": map[string]any{
			"prompt_tokens": res.Usage.PromptTokens, "completion_tokens": res.Usage.CompletionTokens, "total_tokens": res.Usage.TotalTokens,
			"prompt_tokens_details": map[string]any{"cached_tokens": res.Usage.CachedTokens},
		},
		"kolibri_bridge": res,
	}
}

// sseWriter writes a stream on the run's goroutine, the only one that touches it. It sends the headers
// with the first event. A write has a deadline, because no context can interrupt one: a client that
// stops reading would park the turn here, past every other limit, with its request slot. A write that
// fails ends the request through fail, and the writes after it do nothing.
type sseWriter struct {
	b       *Bridge
	w       http.ResponseWriter
	fail    context.CancelCauseFunc
	started bool
	broken  bool
}

func (s *sseWriter) begin() {
	if !s.started {
		s.started = true
		s.w.Header().Set("Content-Type", "text/event-stream")
		s.w.Header().Set("X-Accel-Buffering", "no")
		s.w.WriteHeader(http.StatusOK)
	}
}

func (s *sseWriter) write(text string) {
	if s.broken {
		return
	}
	s.begin()
	s.b.extendWriteDeadline(s.w)
	_, err := io.WriteString(s.w, text)
	if err == nil {
		err = http.NewResponseController(s.w).Flush()
	}
	if err != nil && !errors.Is(err, http.ErrNotSupported) {
		s.broken = true
		s.fail(fmt.Errorf("writing to the client: %w", err))
	}
}

func (s *sseWriter) event(name string, v any) {
	encoded, err := json.Marshal(v)
	if err != nil {
		return
	}
	var b strings.Builder
	if name != "" {
		fmt.Fprintf(&b, "event: %s\n", name)
	}
	fmt.Fprintf(&b, "data: %s\n\n", encoded)
	s.write(b.String())
}

func (s *sseWriter) done() {
	s.write("data: [DONE]\n\n")
}

// refuse answers a refused request. Refusal comes before the first event, so a status still fits.
func (s *sseWriter) refuse(status int, body any) {
	writeJSON(s.w, status, body)
}

func chunkOf(id, model string, delta map[string]any, finish any) map[string]any {
	return map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": time.Now().Unix(), "model": model,
		"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finish}},
	}
}

// serveStream answers like llama-server: chunks of the text, a finish chunk and [DONE], or an error
// event and no [DONE]. The trace goes along as named "kolibri_bridge" events, which a client that
// only looks at unnamed data can ignore. A failure the client should see as a status has to happen
// before the first byte; once the stream is open it is an error event.
func (b *Bridge) serveStream(w http.ResponseWriter, r *http.Request, id string, c *chat) {
	ctx, cancel := context.WithCancelCause(r.Context())
	defer cancel(nil)
	out := &sseWriter{b: b, w: w, fail: cancel}
	res, err := b.start(ctx, id, c, Callbacks{
		Delta: func(d Delta) {
			delta := map[string]any{}
			if d.Content != "" {
				delta["content"] = d.Content
			}
			if d.Reasoning != "" {
				delta["reasoning_content"] = d.Reasoning
			}
			out.event("", chunkOf(d.RequestID, d.Model, delta, nil))
		},
		Event: func(e Event) { out.event("kolibri_bridge", e) },
	})
	var refused *RequestError
	switch {
	case errors.As(err, &refused):
		out.refuse(refused.Status, errorBody(refused.Status, refused.Type, refused.Message))
	case err != nil:
		out.refuse(http.StatusInternalServerError, errorBody(http.StatusInternalServerError, "internal", err.Error()))
	case res.Stop != StopCompleted:
		out.event("", errorBody(statusOf(res.Stop), string(res.Stop), res.Failure.Message))
	default:
		out.event("", chunkOf(res.RequestID, res.Model, map[string]any{}, res.Finish))
		out.done()
	}
}

func (b *Bridge) handleCancel(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var req struct {
		CallID json.RawMessage `json:"call_id"`
	}
	raw, err := b.readBody(w, r, 4<<10)
	if err != nil {
		if b.bodyTooSlow(w, err) {
			return
		}
		writeJSON(w, http.StatusBadRequest, errorBody(http.StatusBadRequest, "invalid_request_error", err.Error()))
		return
	}
	if len(raw) > 0 && json.Unmarshal(raw, &req) != nil {
		writeJSON(w, http.StatusBadRequest, errorBody(http.StatusBadRequest, "invalid_request_error", "the body is not JSON"))
		return
	}
	// Only a body without the field means the whole request. A client whose call ID came out empty
	// wants one operation cancelled, and must not lose the request.
	if req.CallID != nil {
		var callID string
		if err := json.Unmarshal(req.CallID, &callID); err != nil || callID == "" {
			writeJSON(w, http.StatusBadRequest, errorBody(http.StatusBadRequest, "invalid_request_error", "call_id must be a non-empty string; leave it out to cancel the whole request"))
			return
		}
		cancelled, err := b.CancelCall(id, callID)
		if err != nil {
			writeJSON(w, http.StatusNotFound, errorBody(http.StatusNotFound, "not_found", err.Error()))
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"request_id": id, "call_id": callID, "cancelled": cancelled})
		return
	}
	if !b.Cancel(id) {
		writeJSON(w, http.StatusNotFound, errorBody(http.StatusNotFound, "not_found", fmt.Sprintf("no running request %q", id)))
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"request_id": id, "cancelled": true})
}

// Manifest is what the bridge offers: the protocol version, the limits and each operation's schemas.
func (b *Bridge) Manifest() map[string]any {
	var list []map[string]any
	for _, o := range b.reg.Allowed() {
		list = append(list, map[string]any{"name": o.Name, "description": o.Description, "input": o.Input(), "output": o.Output()})
	}
	lim := b.cfg.Limits
	return map[string]any{
		"protocol": ops.Version,
		"limits": map[string]any{
			"max_turns": lim.MaxTurns, "max_calls": lim.MaxCalls, "max_calls_per_turn": lim.MaxCallsPerTurn,
			"max_requests": lim.MaxRequests, "max_running_ops": lim.MaxRunningOps,
			"max_turn_tokens": lim.MaxTurnTokens, "max_turn_bytes": lim.MaxTurnBytes, "max_request_bytes": lim.MaxRequestBytes,
			"op_timeout_ms": lim.OpTimeout.Milliseconds(), "turn_timeout_ms": lim.TurnTimeout.Milliseconds(),
			"request_timeout_ms": lim.RequestTimeout.Milliseconds(), "write_timeout_ms": lim.WriteTimeout.Milliseconds(),
			"body_timeout_ms": lim.BodyTimeout.Milliseconds(),
			"max_arg_bytes":   b.cfg.OpLimits.MaxArgBytes, "max_depth": b.cfg.OpLimits.MaxDepth, "max_int_bits": b.cfg.OpLimits.MaxIntBits,
			"max_array_len": b.cfg.OpLimits.MaxArrayLen, "max_result_bytes": b.cfg.OpLimits.MaxResultBytes,
		},
		"ops": list,
	}
}

func (b *Bridge) handleTools(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, b.Manifest())
}
