package ops

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
)

// Registry is an allowlist of operations under a set of limits. Execute is the only way to run an
// operation. A Registry is safe for concurrent use.
type Registry struct {
	ops     map[string]Op
	allowed map[string]bool
	lim     Limits
}

// NewRegistry returns a registry of the given operations that allows the named ones. A nil allow
// allows them all, an empty one none. A name that isn't an operation is an error, so that a typo in
// an allowlist doesn't pass unnoticed.
func NewRegistry(all []Op, allow []string, lim Limits) (*Registry, error) {
	if lim.MaxArgBytes <= 0 || lim.MaxDepth <= 0 || lim.MaxIntBits <= 0 || lim.MaxArrayLen <= 0 || lim.MaxResultBytes <= 0 {
		return nil, fmt.Errorf("ops: invalid limits %+v", lim)
	}
	r := &Registry{ops: make(map[string]Op, len(all)), allowed: make(map[string]bool, len(all)), lim: lim}
	for _, o := range all {
		if _, dup := r.ops[o.Name]; dup {
			return nil, fmt.Errorf("ops: operation %s registered twice", o.Name)
		}
		r.ops[o.Name] = o
		if allow == nil {
			r.allowed[o.Name] = true
		}
	}
	for _, name := range allow {
		if _, ok := r.ops[name]; !ok {
			return nil, fmt.Errorf("ops: allowlist names unknown operation %q", name)
		}
		r.allowed[name] = true
	}
	return r, nil
}

// Limits returns the registry's limits.
func (r *Registry) Limits() Limits { return r.lim }

// Failure returns the response of a request that the registry didn't get to answer, because its
// deadline passed or it was cancelled.
func (r *Registry) Failure(id, op string, code Code, message string) Response {
	resp := Response{Version: Version, ID: id, Error: &Error{Code: code, Message: message}}
	if _, known := r.ops[op]; known {
		resp.Op = op
	}
	return resp
}

// Allowed lists the allowed operations, sorted by name.
func (r *Registry) Allowed() []Op {
	var out []Op
	for name := range r.allowed {
		out = append(out, r.ops[name])
	}
	slices.SortFunc(out, func(x, y Op) int {
		switch {
		case x.Name < y.Name:
			return -1
		case x.Name > y.Name:
			return 1
		}
		return 0
	})
	return out
}

// Tools returns the allowed operations as OpenAI tools, in a fixed order and encoding, so that the
// rendered prompt doesn't vary between runs.
func (r *Registry) Tools() []json.RawMessage {
	allowed := r.Allowed()
	out := make([]json.RawMessage, 0, len(allowed))
	for _, o := range allowed {
		name, _ := json.Marshal(o.Name)
		desc, _ := json.Marshal(o.Description)
		tool := fmt.Sprintf(`{"type":"function","function":{"name":%s,"description":%s,"parameters":%s}}`, name, desc, o.Input())
		out = append(out, json.RawMessage(tool))
	}
	return out
}

// Execute runs a request and always returns a response. Failures come back as a Response with
// OK false: the model is told what went wrong and can try again.
func (r *Registry) Execute(ctx context.Context, req Request) Response {
	resp := Response{Version: Version, ID: req.ID}
	op, known := r.ops[req.Op]
	if known {
		resp.Op = op.Name
	}
	result, err := r.run(ctx, op, known, req)
	if err != nil {
		resp.Error = err
		return resp
	}
	resp.OK = true
	resp.Result = result
	return resp
}

func (r *Registry) run(ctx context.Context, op Op, known bool, req Request) (result json.RawMessage, opErr *Error) {
	defer func() {
		if p := recover(); p != nil {
			result, opErr = nil, errorf(CodeInternal, "operation %s panicked: %v", req.Op, p)
		}
	}()
	switch {
	case !known:
		return nil, errorf(CodeUnknownOp, "no operation %q", clip(req.Op))
	case !r.allowed[req.Op]:
		return nil, errorf(CodeOpNotAllowed, "operation %q is not allowed", req.Op)
	case len(req.Args) > r.lim.MaxArgBytes:
		return nil, errorf(CodeInputTooLarge, "the arguments have %d bytes, the limit is %d", len(req.Args), r.lim.MaxArgBytes)
	}
	fields, opErr := checkArgs(req.Args, r.lim.MaxDepth)
	if opErr != nil {
		return nil, opErr
	}
	if v, ok := fields["v"]; ok {
		// A request may name the protocol version it means. No operation has a field v.
		if string(v) != fmt.Sprint(Version) {
			return nil, errorf(CodeUnsupportedVersion, "this bridge speaks version %d, the request names %s", Version, clip(string(v)))
		}
		delete(fields, "v")
	}
	if ctx.Err() != nil {
		return nil, errorf(CodeCancelled, "cancelled before the operation started")
	}
	result, opErr = op.run(ctx, r.lim, fields)
	if opErr != nil {
		return nil, opErr
	}
	if len(result) > r.lim.MaxResultBytes {
		return nil, errorf(CodeResultTooLarge, "the result has %d bytes, the limit is %d", len(result), r.lim.MaxResultBytes)
	}
	return result, nil
}
