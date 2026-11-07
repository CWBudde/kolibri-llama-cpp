package ops

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math/big"
	"strconv"
	"strings"
	"unicode/utf8"
)

// checkArgs accepts exactly one JSON object: valid UTF-8, no duplicate key at any level, nesting up
// to maxDepth, nothing after it. encoding/json alone takes the last of two equal keys and replaces
// invalid UTF-8, so {"a":"1","a":"2"} would run on a value the model may not have meant. It returns
// the object's top-level fields.
func checkArgs(raw []byte, maxDepth int) (map[string]json.RawMessage, *Error) {
	if !utf8.Valid(raw) {
		return nil, errorf(CodeMalformedArguments, "arguments are not valid UTF-8")
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	type level struct {
		object bool
		key    bool // the next token of the object is a key
		seen   map[string]bool
	}
	var stack []level
	for {
		tok, err := dec.Token()
		if errors.Is(err, io.EOF) {
			if len(stack) != 0 {
				return nil, errorf(CodeMalformedArguments, "arguments end inside a value")
			}
			break
		}
		if err != nil {
			return nil, errorf(CodeMalformedArguments, "arguments are not valid JSON: %v", err)
		}
		if len(stack) > 0 && stack[len(stack)-1].object && stack[len(stack)-1].key {
			if key, ok := tok.(string); ok {
				top := &stack[len(stack)-1]
				if top.seen[key] {
					return nil, errorf(CodeMalformedArguments, "duplicate key %q", clip(key))
				}
				top.seen[key] = true
				top.key = false
				continue
			}
		}
		switch tok {
		case json.Delim('{'), json.Delim('['):
			if len(stack) == maxDepth {
				return nil, errorf(CodeMalformedArguments, "arguments nest deeper than %d", maxDepth)
			}
			if len(stack) > 0 && stack[len(stack)-1].object {
				stack[len(stack)-1].key = true
			}
			object := tok == json.Delim('{')
			stack = append(stack, level{object: object, key: object, seen: map[string]bool{}})
			continue
		case json.Delim('}'), json.Delim(']'):
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				if _, err := dec.Token(); !errors.Is(err, io.EOF) {
					return nil, errorf(CodeMalformedArguments, "data after the arguments object")
				}
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(raw, &fields); err != nil {
					return nil, errorf(CodeMalformedArguments, "arguments are not a JSON object")
				}
				return fields, nil
			}
			continue
		}
		if len(stack) == 0 {
			return nil, errorf(CodeMalformedArguments, "arguments are not a JSON object")
		}
		if stack[len(stack)-1].object {
			stack[len(stack)-1].key = true
		}
	}
	return nil, errorf(CodeMalformedArguments, "arguments are empty")
}

// clip bounds model-written text that goes into an error message.
func clip(s string) string {
	const limit = 64
	if len(s) <= limit {
		return s
	}
	for i := limit; i > 0; i-- {
		if utf8.RuneStart(s[i]) {
			return s[:i] + "..."
		}
	}
	return "..."
}

// decodeInto checks the fields against the operation's input schema, then decodes them. Each
// field's own type does the rest. null is refused here because encoding/json would otherwise turn
// it into a zero value without a word.
func decodeInto(fields map[string]json.RawMessage, schema inputSchema, in any) *Error {
	for _, name := range schema.required {
		if _, ok := fields[name]; !ok {
			return errorf(CodeInvalidArgument, "missing field %q", name)
		}
	}
	known := map[string]bool{}
	for _, p := range schema.props {
		known[p.name] = true
	}
	merged := make(map[string]json.RawMessage, len(fields))
	for name, value := range fields {
		if !known[name] {
			return errorf(CodeInvalidArgument, "unknown field %q", clip(name))
		}
		if string(bytes.TrimSpace(value)) == "null" {
			return errorf(CodeInvalidArgument, "field %q is null", name)
		}
		merged[name] = value
	}
	encoded, err := json.Marshal(merged)
	if err != nil {
		return errorf(CodeInternal, "re-encoding the arguments: %v", err)
	}
	dec := json.NewDecoder(bytes.NewReader(encoded))
	dec.UseNumber()
	if err := dec.Decode(in); err != nil {
		var opErr *Error
		if errors.As(err, &opErr) {
			return opErr
		}
		var typeErr *json.UnmarshalTypeError
		if errors.As(err, &typeErr) {
			return errorf(CodeInvalidArgument, "field %q: expected %s, got %s", typeErr.Field, typeErr.Type, typeErr.Value)
		}
		return errorf(CodeInvalidArgument, "%v", err)
	}
	return nil
}

// intArg is an integer input, unparsed until the limits are known. It takes a decimal string or a
// JSON integer. A JSON number with a fraction or an exponent is refused even if its value is whole,
// and so is one outside int64: llama-server parses the arguments when it renders the next prompt,
// and its JSON library turns 2^63 into -2^63 and 10^23 into 1e+23, so the model would read a number
// other than the one that was computed with. The tool schema asks for strings for the same reason.
type intArg string

// UnmarshalJSON implements json.Unmarshaler.
func (a *intArg) UnmarshalJSON(raw []byte) error {
	switch {
	case len(raw) >= 2 && raw[0] == '"':
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return errorf(CodeInvalidArgument, "bad string: %v", err)
		}
		*a = intArg(s)
		return nil
	case len(raw) > 0 && (raw[0] == '-' || (raw[0] >= '0' && raw[0] <= '9')):
		s := string(raw)
		if _, err := strconv.ParseInt(s, 10, 64); err != nil {
			if strings.ContainsAny(s, ".eE") {
				return errorf(CodeInvalidArgument, "%s is not an integer: only integers are supported", clip(s))
			}
			return errorf(CodeInvalidArgument, "%s is outside int64: send it as a decimal string", clip(s))
		}
		*a = intArg(s)
		return nil
	}
	return errorf(CodeInvalidArgument, "expected an integer as a decimal string, got %s", clip(string(raw)))
}

// parse returns the integer, or an error that names the field. The syntax is the schema's pattern:
// an optional minus and digits without a leading zero, and no "-0". The length check comes first
// because parsing is superlinear. A minus and one digit per 3.32 bits fit in the bound.
func (a intArg) parse(field string, lim Limits) (*big.Int, *Error) {
	s := string(a)
	if len(s) > lim.MaxIntBits/3+2 {
		return nil, errorf(CodeInputTooLarge, "%s has %d characters, the limit is %d bits", field, len(s), lim.MaxIntBits)
	}
	digits := strings.TrimPrefix(s, "-")
	if digits == "" || (len(digits) > 1 && digits[0] == '0') || s == "-0" || strings.Trim(digits, "0123456789") != "" {
		return nil, errorf(CodeInvalidArgument, "%s: %q is not a decimal integer", field, clip(s))
	}
	v, ok := new(big.Int).SetString(s, 10)
	if !ok {
		return nil, errorf(CodeInvalidArgument, "%s: %q is not a decimal integer", field, clip(s))
	}
	if v.BitLen() > lim.MaxIntBits {
		return nil, errorf(CodeInputTooLarge, "%s has %d bits, the limit is %d", field, v.BitLen(), lim.MaxIntBits)
	}
	return v, nil
}

// checked returns v as a result, or overflow past the limit. Every integer result passes here.
func checked(v *big.Int, lim Limits) (string, *Error) {
	if v.BitLen() > lim.MaxIntBits {
		return "", errorf(CodeOverflow, "the result has %d bits, the limit is %d", v.BitLen(), lim.MaxIntBits)
	}
	return v.String(), nil
}
