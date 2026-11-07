// Package ops is the execution side of the bridge: protocol version 1 and a registry of
// deterministic operations. Its code has no time, randomness, file or network access, which
// TestImportsStayPure enforces. Nothing in it knows about a model or HTTP.
//
// An operation's input is a JSON object, its output a typed JSON object, and the same request gives
// the same bytes every time. Integers are exact: decimal strings on both sides, arbitrary precision
// up to Limits.MaxIntBits. A result past that limit is the error "overflow", never a rounded value.
// Non-integers are rejected, there is no floating point here.
package ops

import (
	"encoding/json"
	"fmt"
)

// Version is the protocol version. It changes when a response or an operation's meaning changes.
const Version = 1

// Code is a machine-readable error. The values are part of the protocol.
type Code string

const (
	CodeMalformedArguments Code = "malformed_arguments" // the arguments aren't one valid JSON object
	CodeUnknownOp          Code = "unknown_op"          // no such operation
	CodeOpNotAllowed       Code = "op_not_allowed"      // it exists, the allowlist excludes it
	CodeUnsupportedVersion Code = "unsupported_version" // the request names another protocol version
	CodeInvalidArgument    Code = "invalid_argument"    // valid JSON, wrong field, type or value
	CodeInputTooLarge      Code = "input_too_large"     // an input is past a limit
	CodeOverflow           Code = "overflow"            // an integer result is past MaxIntBits
	CodeDivisionByZero     Code = "division_by_zero"    // the divisor is 0
	CodeResultTooLarge     Code = "result_too_large"    // the encoded result is past its limit
	CodeTimeout            Code = "timeout"             // the operation ran past its deadline
	CodeCancelled          Code = "cancelled"           // the operation or its request was cancelled
	CodeInternal           Code = "internal"            // a bug here, for example a panic in an operation
)

// Error is a failed operation, as the model and the client see it.
type Error struct {
	Code    Code   `json:"code"`
	Message string `json:"message"`
}

func (e *Error) Error() string { return string(e.Code) + ": " + e.Message }

func errorf(code Code, format string, args ...any) *Error {
	return &Error{Code: code, Message: fmt.Sprintf(format, args...)}
}

// Request is one execution request. Args is the operation's input as raw JSON text, unparsed, so
// that the size limit and the strict decoder see what the model wrote.
type Request struct {
	ID   string // correlates the response; the bridge assigns it
	Op   string
	Args []byte
}

// Response is the result of a request, the only thing that goes back to the model. OK decides
// which of Result and Error is set. Op is set only when it names a registered operation, so that a
// model can't echo arbitrary text through it.
type Response struct {
	Version int             `json:"v"`
	ID      string          `json:"id"`
	Op      string          `json:"op,omitempty"`
	OK      bool            `json:"ok"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *Error          `json:"error,omitempty"`
}

// Encode returns the response's canonical bytes: the field order above, compact. encoding/json's
// HTML escaping stays on on purpose: an error message can quote model-written text, and escaped it
// can't close the template's <tool_response> block. The same response always encodes to the same
// bytes.
func (r Response) Encode() string {
	b, err := json.Marshal(r)
	if err != nil { // only an invalid Result can do this, and a Result comes from json.Marshal
		panic(fmt.Sprintf("ops: response of %s: %v", r.ID, err))
	}
	return string(b)
}

// Limits bounds the work of one operation. Operations can't be interrupted inside a math/big call,
// so these limits, checked before the work starts, are what bounds their time and memory.
type Limits struct {
	MaxArgBytes    int // the raw arguments
	MaxDepth       int // JSON nesting of the arguments
	MaxIntBits     int // an integer input or result
	MaxArrayLen    int // an array input
	MaxResultBytes int // the encoded result
}

// DefaultLimits returns the limits of a default bridge.
func DefaultLimits() Limits {
	return Limits{
		MaxArgBytes:    256 << 10,
		MaxDepth:       8,
		MaxIntBits:     4096, // 1234 decimal digits
		MaxArrayLen:    4096,
		MaxResultBytes: 1 << 20,
	}
}
