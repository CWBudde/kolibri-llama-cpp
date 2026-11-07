package bridge

import (
	"crypto/sha256"
	"encoding/hex"
	"slices"
)

// EventKind names a step of a request.
type EventKind string

const (
	EventRequestStart    EventKind = "request_start"
	EventTurnStart       EventKind = "turn_start"
	EventTurnEnd         EventKind = "turn_end"
	EventCallRefused     EventKind = "call_refused" // the turn's outcome forbids running it
	EventCallStart       EventKind = "call_start"
	EventCallEnd         EventKind = "call_end"
	EventCancelRequested EventKind = "cancel_requested"
	EventRequestEnd      EventKind = "request_end"
)

// Event is one entry of a request's trace. The text fields are clipped at traceTextLimit, the hashes
// always cover the whole text.
type Event struct {
	Seq       int       `json:"seq"`
	Kind      EventKind `json:"kind"`
	RequestID string    `json:"request_id"`

	Protocol int      `json:"protocol,omitempty"` // request_start
	Allowed  []string `json:"allowed,omitempty"`  // request_start

	Turn       int    `json:"turn,omitempty"`
	BodySHA256 string `json:"body_sha256,omitempty"` // turn_start: the exact bytes sent to the server
	Finish     string `json:"finish,omitempty"`      // turn_end
	Calls      int    `json:"calls,omitempty"`       // turn_end
	Content    string `json:"content,omitempty"`     // turn_end
	Reasoning  string `json:"reasoning,omitempty"`   // turn_end

	CallID       string `json:"call_id,omitempty"`
	Op           string `json:"op,omitempty"`
	Args         string `json:"args,omitempty"`
	ArgsSHA256   string `json:"args_sha256,omitempty"`
	Result       string `json:"result,omitempty"` // call_end: the tool message as the model reads it
	ResultSHA256 string `json:"result_sha256,omitempty"`
	JobState     string `json:"job_state,omitempty"` // call_end: the scheduler's state of the job
	Code         string `json:"code,omitempty"`      // call_end: the operation's error code

	Stop    StopReason `json:"stop,omitempty"`    // request_end
	Failure *Failure   `json:"failure,omitempty"` // request_end

	// These differ between identical runs: Canonical clears them.
	ElapsedMS  int64  `json:"elapsed_ms,omitempty"`
	UpstreamID string `json:"upstream_id,omitempty"` // call_start: the server's random call ID
	Usage      *Usage `json:"usage,omitempty"`       // turn_end, request_end: cached tokens depend on the slot
}

const traceTextLimit = 4 << 10

func clipText(s string) string {
	if len(s) <= traceTextLimit {
		return s
	}
	cut := traceTextLimit
	for cut > 0 && s[cut]&0xC0 == 0x80 { // not in the middle of a UTF-8 sequence
		cut--
	}
	return s[:cut] + "..."
}

func sha256Hex(s []byte) string {
	sum := sha256.Sum256(s)
	return hex.EncodeToString(sum[:])
}

// Canonical returns a trace without what legitimately differs between two identical runs. Two runs
// of one request against one script have the same canonical trace.
func Canonical(events []Event) []Event {
	out := slices.Clone(events)
	for i := range out {
		out[i].ElapsedMS = 0
		out[i].UpstreamID = ""
		out[i].Usage = nil
	}
	return out
}
