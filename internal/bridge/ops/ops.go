package ops

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"slices"
	"strings"
)

// prop is one field of an operation's input or output.
type prop struct {
	name   string
	schema string // a JSON schema
	doc    string
}

// inputSchema is an operation's input object. Fields that aren't required are optional.
type inputSchema struct {
	props    []prop
	required []string
}

// A JSON schema for an integer is a string with a pattern, not {"type":"integer"}: llama.cpp's
// grammar for that allows 16 digits (common/json-schema-to-grammar.cpp, "integral-part"), so a model
// under the tool grammar couldn't write a longer number and would have to end it early.
const (
	intSchema     = `{"type":"string","pattern":"^(0|-?[1-9][0-9]*)$"}`
	intListSchema = `{"type":"array","items":` + intSchema + `}`
	strSchema     = `{"type":"string"}`
	boolSchema    = `{"type":"boolean"}`
	hashEncSchema = `{"type":"string","enum":["utf8","hex","base64"]}`
)

func (s inputSchema) json() json.RawMessage { return objectSchema(s.props, s.required, true) }

func objectSchema(props []prop, required []string, closed bool) json.RawMessage {
	var b bytes.Buffer
	b.WriteString(`{"type":"object","properties":{`)
	for i, p := range props {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%q:", p.name)
		if p.doc == "" {
			b.WriteString(p.schema)
		} else {
			// The description goes first so that the schema reads the same whatever it holds.
			b.WriteString(`{"description":`)
			doc, _ := json.Marshal(p.doc)
			b.Write(doc)
			b.WriteByte(',')
			b.WriteString(strings.TrimPrefix(p.schema, "{"))
		}
	}
	b.WriteString(`},"required":[`)
	for i, name := range required {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%q", name)
	}
	b.WriteString(`]`)
	if closed {
		b.WriteString(`,"additionalProperties":false`)
	}
	b.WriteString(`}`)
	return b.Bytes()
}

// Op is a registered operation. Run gets the arguments only after the registry's checks.
type Op struct {
	Name        string
	Description string
	input       inputSchema
	output      []prop
	run         func(ctx context.Context, lim Limits, fields map[string]json.RawMessage) (json.RawMessage, *Error)
}

// Input returns the operation's input as a JSON schema.
func (o Op) Input() json.RawMessage { return o.input.json() }

// Output returns the operation's output as a JSON schema.
func (o Op) Output() json.RawMessage {
	var required []string
	for _, p := range o.output {
		required = append(required, p.name)
	}
	return objectSchema(o.output, required, true)
}

// define builds an Op from a typed function, so that every operation decodes its input the same way.
func define[In, Out any](name, desc string, input inputSchema, output []prop,
	run func(context.Context, Limits, In) (Out, *Error)) Op {
	return Op{
		Name: name, Description: desc, input: input, output: output,
		run: func(ctx context.Context, lim Limits, fields map[string]json.RawMessage) (json.RawMessage, *Error) {
			var in In
			if err := decodeInto(fields, input, &in); err != nil {
				return nil, err
			}
			out, opErr := run(ctx, lim, in)
			if opErr != nil {
				return nil, opErr
			}
			encoded, err := json.Marshal(out)
			if err != nil {
				return nil, errorf(CodeInternal, "encoding the result: %v", err)
			}
			return encoded, nil
		},
	}
}

// DefineForTest makes an operation with an arbitrary body, for tests in other packages: a slow
// operation is how a deadline gets tested, and no real operation is slow. Builtin never returns one.
func DefineForTest(name string, run func(ctx context.Context, args json.RawMessage) (json.RawMessage, *Error)) Op {
	return Op{
		Name: name, Description: "test operation",
		run: func(ctx context.Context, _ Limits, fields map[string]json.RawMessage) (json.RawMessage, *Error) {
			args, err := json.Marshal(fields)
			if err != nil {
				return nil, errorf(CodeInternal, "%v", err)
			}
			return run(ctx, args)
		},
	}
}

// Failure builds an Error, for a test operation.
func Failure(code Code, message string) *Error { return &Error{Code: code, Message: message} }

type twoInts struct {
	A intArg `json:"a"`
	B intArg `json:"b"`
}

type intValue struct {
	Value string `json:"value"`
}

type intList struct {
	Values []intArg `json:"values"`
}

var (
	abInput   = inputSchema{props: []prop{{name: "a", schema: intSchema}, {name: "b", schema: intSchema}}, required: []string{"a", "b"}}
	listInput = inputSchema{props: []prop{{name: "values", schema: intListSchema}}, required: []string{"values"}}
	valueOut  = []prop{{name: "value", schema: intSchema}}
	listOut   = []prop{{name: "values", schema: intListSchema}}
	intNote   = "Integers are decimal strings, exact up to the size limit."
)

func (in twoInts) parse(lim Limits) (a, b *big.Int, err *Error) {
	if a, err = in.A.parse("a", lim); err != nil {
		return nil, nil, err
	}
	if b, err = in.B.parse("b", lim); err != nil {
		return nil, nil, err
	}
	return a, b, nil
}

func binary(name, desc string, f func(a, b *big.Int) *big.Int) Op {
	return define(name, desc+" "+intNote, abInput, valueOut,
		func(_ context.Context, lim Limits, in twoInts) (intValue, *Error) {
			a, b, err := in.parse(lim)
			if err != nil {
				return intValue{}, err
			}
			s, err := checked(f(a, b), lim)
			return intValue{s}, err
		})
}

// parseList bounds the array before it parses an element.
func (in intList) parse(ctx context.Context, lim Limits) ([]*big.Int, *Error) {
	if len(in.Values) > lim.MaxArrayLen {
		return nil, errorf(CodeInputTooLarge, "values has %d elements, the limit is %d", len(in.Values), lim.MaxArrayLen)
	}
	out := make([]*big.Int, len(in.Values))
	for i, v := range in.Values {
		if err := ctx.Err(); err != nil {
			return nil, errorf(CodeCancelled, "cancelled")
		}
		var err *Error
		if out[i], err = v.parse(fmt.Sprintf("values[%d]", i), lim); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func fold(name, desc string, start int64, step func(acc, v *big.Int)) Op {
	return define(name, desc+" "+intNote, listInput, valueOut,
		func(ctx context.Context, lim Limits, in intList) (intValue, *Error) {
			values, err := in.parse(ctx, lim)
			if err != nil {
				return intValue{}, err
			}
			acc := big.NewInt(start)
			for _, v := range values {
				if ctx.Err() != nil {
					return intValue{}, errorf(CodeCancelled, "cancelled")
				}
				step(acc, v)
				if acc.BitLen() > lim.MaxIntBits { // stop here, a product would keep growing
					return intValue{}, errorf(CodeOverflow, "the result passes %d bits", lim.MaxIntBits)
				}
			}
			s, err := checked(acc, lim)
			return intValue{s}, err
		})
}

func extreme(name, desc string, sign int) Op {
	return define(name, desc+" An empty array is an invalid_argument. "+intNote, listInput, valueOut,
		func(ctx context.Context, lim Limits, in intList) (intValue, *Error) {
			values, err := in.parse(ctx, lim)
			if err != nil {
				return intValue{}, err
			}
			if len(values) == 0 {
				return intValue{}, errorf(CodeInvalidArgument, "values is empty")
			}
			best := values[0]
			for _, v := range values[1:] {
				if v.Cmp(best)*sign > 0 {
					best = v
				}
			}
			s, err := checked(best, lim)
			return intValue{s}, err
		})
}

func listOp(name, desc string, input inputSchema, output []prop, f func(values []*big.Int, descending bool)) Op {
	type in struct {
		Values     []intArg `json:"values"`
		Descending bool     `json:"descending"`
	}
	type out struct {
		Values []string `json:"values"`
	}
	return define(name, desc+" "+intNote, input, output,
		func(ctx context.Context, lim Limits, v in) (out, *Error) {
			values, err := intList{Values: v.Values}.parse(ctx, lim)
			if err != nil {
				return out{}, err
			}
			f(values, v.Descending)
			result := make([]string, len(values)) // not nil: an empty list encodes as []
			for i, x := range values {
				result[i] = x.String()
			}
			return out{result}, nil
		})
}

type divResult struct {
	Quotient  string `json:"quotient"`
	Remainder string `json:"remainder"`
}

type cmpResult struct {
	Order int `json:"order"`
}

type powInput struct {
	Base     intArg `json:"base"`
	Exponent intArg `json:"exponent"`
}

type hashInput struct {
	Data     string  `json:"data"`
	Encoding *string `json:"encoding"` // nil if absent; empty is not in the schema's enum
}

type hashResult struct {
	Hex string `json:"hex"`
}

func hashOp(name, desc string, sum func([]byte) []byte) Op {
	input := inputSchema{
		props: []prop{
			{name: "data", schema: strSchema},
			{name: "encoding", schema: hashEncSchema, doc: "how data is written, utf8 if absent"},
		},
		required: []string{"data"},
	}
	return define(name, desc, input, []prop{{name: "hex", schema: strSchema}},
		func(_ context.Context, _ Limits, in hashInput) (hashResult, *Error) {
			encoding := "utf8"
			if in.Encoding != nil {
				encoding = *in.Encoding
			}
			var data []byte
			var err error
			switch encoding {
			case "utf8":
				data = []byte(in.Data)
			case "hex":
				data, err = hex.DecodeString(in.Data)
			case "base64":
				// Strict refuses non-zero trailing bits. The decoder skips line breaks even then, and
				// two different strings shouldn't be one input without an error.
				if strings.ContainsAny(in.Data, "\r\n") {
					return hashResult{}, errorf(CodeInvalidArgument, "data is not valid base64: it has a line break")
				}
				data, err = base64.StdEncoding.Strict().DecodeString(in.Data)
			default:
				return hashResult{}, errorf(CodeInvalidArgument, "unknown encoding %q", clip(encoding))
			}
			if err != nil {
				return hashResult{}, errorf(CodeInvalidArgument, "data is not valid %s: %v", encoding, err)
			}
			return hashResult{hex.EncodeToString(sum(data))}, nil
		})
}

// Builtin returns every operation, sorted by name.
func Builtin() []Op {
	sortInput := inputSchema{
		props:    []prop{{name: "values", schema: intListSchema}, {name: "descending", schema: boolSchema, doc: "false if absent"}},
		required: []string{"values"},
	}
	all := []Op{
		binary("int_add", "a + b.", func(a, b *big.Int) *big.Int { return a.Add(a, b) }),
		binary("int_sub", "a - b.", func(a, b *big.Int) *big.Int { return a.Sub(a, b) }),
		binary("int_mul", "a * b.", func(a, b *big.Int) *big.Int { return a.Mul(a, b) }),
		binary("int_gcd", "Greatest common divisor of |a| and |b|; gcd(0,0) is 0.", func(a, b *big.Int) *big.Int {
			return new(big.Int).GCD(nil, nil, new(big.Int).Abs(a), new(big.Int).Abs(b))
		}),
		define("int_div", "a / b, truncated toward zero, and the remainder, which has a's sign, so a = quotient*b + remainder. "+intNote,
			abInput, []prop{{name: "quotient", schema: intSchema}, {name: "remainder", schema: intSchema}},
			func(_ context.Context, lim Limits, in twoInts) (divResult, *Error) {
				a, b, err := in.parse(lim)
				if err != nil {
					return divResult{}, err
				}
				if b.Sign() == 0 {
					return divResult{}, errorf(CodeDivisionByZero, "the divisor is 0")
				}
				q, r := new(big.Int).QuoRem(a, b, new(big.Int))
				return divResult{q.String(), r.String()}, nil
			}),
		define("int_pow", "base to the power exponent, for an exponent >= 0; 0 to the 0 is 1. "+intNote,
			inputSchema{props: []prop{{name: "base", schema: intSchema}, {name: "exponent", schema: intSchema}}, required: []string{"base", "exponent"}},
			valueOut, pow),
		define("int_cmp", "Order of a and b: -1 if a < b, 0 if equal, 1 if a > b. "+intNote,
			abInput, []prop{{name: "order", schema: `{"type":"integer","enum":[-1,0,1]}`}},
			func(_ context.Context, lim Limits, in twoInts) (cmpResult, *Error) {
				a, b, err := in.parse(lim)
				if err != nil {
					return cmpResult{}, err
				}
				return cmpResult{a.Cmp(b)}, nil
			}),
		fold("array_sum", "Sum of values; of an empty array, 0.", 0, func(acc, v *big.Int) { acc.Add(acc, v) }),
		fold("array_product", "Product of values; of an empty array, 1.", 1, func(acc, v *big.Int) { acc.Mul(acc, v) }),
		extreme("array_min", "Smallest of values.", -1),
		extreme("array_max", "Largest of values.", 1),
		listOp("array_sort", "values in numeric order.", sortInput, listOut, func(values []*big.Int, descending bool) {
			slices.SortStableFunc(values, func(x, y *big.Int) int {
				if descending {
					return y.Cmp(x)
				}
				return x.Cmp(y)
			})
		}),
		listOp("array_reverse", "values in reverse order.", listInput, listOut, func(values []*big.Int, _ bool) {
			slices.Reverse(values)
		}),
		hashOp("hash_sha256", "SHA-256 of data, in lower-case hex.", func(b []byte) []byte { s := sha256.Sum256(b); return s[:] }),
		hashOp("hash_sha512", "SHA-512 of data, in lower-case hex.", func(b []byte) []byte { s := sha512.Sum512(b); return s[:] }),
	}
	slices.SortFunc(all, func(x, y Op) int { return strings.Compare(x.Name, y.Name) })
	return all
}

// pow checks the size before it does the work. The result of |base| >= 2 has more than
// (bitlen(base)-1)*exponent bits, so a result past the limit is known without computing it, and the
// exponent of a result within it has at most MaxIntBits bits' worth of squarings.
func pow(_ context.Context, lim Limits, in powInput) (intValue, *Error) {
	base, err := in.Base.parse("base", lim)
	if err != nil {
		return intValue{}, err
	}
	exp, err := in.Exponent.parse("exponent", lim)
	if err != nil {
		return intValue{}, err
	}
	if exp.Sign() < 0 {
		return intValue{}, errorf(CodeInvalidArgument, "exponent is negative, the result would not be an integer")
	}
	abs := new(big.Int).Abs(base)
	switch {
	case abs.Sign() == 0 || abs.Cmp(big.NewInt(1)) == 0:
		// 0 and ±1 stay small for any exponent, and a huge exponent must not reach Exp's loop.
		if exp.Sign() == 0 {
			return intValue{"1"}, nil
		}
		if base.Sign() < 0 && exp.Bit(0) == 0 {
			return intValue{"1"}, nil
		}
		return intValue{base.String()}, nil
	case !exp.IsInt64() || exp.Int64() > int64(lim.MaxIntBits):
		return intValue{}, errorf(CodeOverflow, "base^exponent passes %d bits", lim.MaxIntBits)
	case int64(abs.BitLen()-1)*exp.Int64() >= int64(lim.MaxIntBits):
		return intValue{}, errorf(CodeOverflow, "base^exponent passes %d bits", lim.MaxIntBits)
	}
	s, err := checked(new(big.Int).Exp(base, exp, nil), lim)
	return intValue{s}, err
}
