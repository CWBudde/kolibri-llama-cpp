package ops

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go/parser"
	"go/token"
	"math/big"
	"math/rand/v2"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
)

func registry(t *testing.T) *Registry {
	t.Helper()
	r, err := NewRegistry(Builtin(), nil, DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func run(t *testing.T, r *Registry, op, args string) Response {
	t.Helper()
	return r.Execute(context.Background(), Request{ID: "x", Op: op, Args: []byte(args)})
}

// wantResult checks the exact result bytes.
func wantResult(t *testing.T, r *Registry, op, args, want string) {
	t.Helper()
	resp := run(t, r, op, args)
	if !resp.OK || string(resp.Result) != want {
		t.Errorf("%s(%s) = %s, want result %s", op, args, resp.Encode(), want)
	}
}

func wantCode(t *testing.T, r *Registry, op, args string, code Code) {
	t.Helper()
	resp := run(t, r, op, args)
	if resp.OK || resp.Error == nil || resp.Error.Code != code {
		t.Errorf("%s(%s) = %s, want error %s", op, args, resp.Encode(), code)
	}
}

// Every value here is worked out by hand, or by bc for the long ones, not by math/big.
func TestExactResults(t *testing.T) {
	r := registry(t)
	for _, c := range []struct{ op, args, want string }{
		{"int_add", `{"a":"2","b":"3"}`, `{"value":"5"}`},
		{"int_add", `{"a":"-2","b":"2"}`, `{"value":"0"}`},
		{"int_add", `{"a":"99999999999999999999999999999999","b":"1"}`, `{"value":"100000000000000000000000000000000"}`},
		{"int_sub", `{"a":"2","b":"3"}`, `{"value":"-1"}`},
		{"int_sub", `{"a":"100000000000000000000000000000000","b":"1"}`, `{"value":"99999999999999999999999999999999"}`},
		{"int_mul", `{"a":"987654321987654321","b":"123456789123456789"}`, `{"value":"121932631356500531347203169112635269"}`},
		{"int_mul", `{"a":"-3","b":"7"}`, `{"value":"-21"}`},
		{"int_mul", `{"a":"0","b":"-7"}`, `{"value":"0"}`},
		{"int_div", `{"a":"7","b":"2"}`, `{"quotient":"3","remainder":"1"}`},
		{"int_div", `{"a":"-7","b":"2"}`, `{"quotient":"-3","remainder":"-1"}`},
		{"int_div", `{"a":"7","b":"-2"}`, `{"quotient":"-3","remainder":"1"}`},
		{"int_div", `{"a":"-7","b":"-2"}`, `{"quotient":"3","remainder":"-1"}`},
		{"int_div", `{"a":"0","b":"5"}`, `{"quotient":"0","remainder":"0"}`},
		{"int_pow", `{"base":"2","exponent":"10"}`, `{"value":"1024"}`},
		{"int_pow", `{"base":"-2","exponent":"3"}`, `{"value":"-8"}`},
		{"int_pow", `{"base":"0","exponent":"0"}`, `{"value":"1"}`},
		{"int_pow", `{"base":"7","exponent":"0"}`, `{"value":"1"}`},
		{"int_pow", `{"base":"0","exponent":"1000000000000000000000000"}`, `{"value":"0"}`},
		{"int_pow", `{"base":"1","exponent":"1000000000000000000000000"}`, `{"value":"1"}`},
		{"int_pow", `{"base":"-1","exponent":"1000000000000000000000001"}`, `{"value":"-1"}`},
		{"int_pow", `{"base":"-1","exponent":"1000000000000000000000000"}`, `{"value":"1"}`},
		{"int_gcd", `{"a":"12","b":"-18"}`, `{"value":"6"}`},
		{"int_gcd", `{"a":"0","b":"0"}`, `{"value":"0"}`},
		{"int_cmp", `{"a":"-1","b":"1"}`, `{"order":-1}`},
		{"int_cmp", `{"a":"1","b":"1"}`, `{"order":0}`},
		{"int_cmp", `{"a":"100000000000000000000","b":"99999999999999999999"}`, `{"order":1}`},
		{"array_sum", `{"values":["1","-2","3"]}`, `{"value":"2"}`},
		{"array_sum", `{"values":[]}`, `{"value":"0"}`},
		{"array_product", `{"values":["2","-3","4"]}`, `{"value":"-24"}`},
		{"array_product", `{"values":[]}`, `{"value":"1"}`},
		{"array_min", `{"values":["3","-5","4"]}`, `{"value":"-5"}`},
		{"array_max", `{"values":["3","-5","4"]}`, `{"value":"4"}`},
		{"array_sort", `{"values":["10","9","-1","9"]}`, `{"values":["-1","9","9","10"]}`},
		{"array_sort", `{"values":["10","9","-1"],"descending":true}`, `{"values":["10","9","-1"]}`},
		{"array_sort", `{"values":[]}`, `{"values":[]}`},
		{"array_reverse", `{"values":["1","2","3"]}`, `{"values":["3","2","1"]}`},
		{"hash_sha256", `{"data":"abc"}`, `{"hex":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}`},
		{"hash_sha256", `{"data":""}`, `{"hex":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}`},
		{"hash_sha256", `{"data":"616263","encoding":"hex"}`, `{"hex":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}`},
		{"hash_sha256", `{"data":"YWJj","encoding":"base64"}`, `{"hex":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}`},
		{"hash_sha256", `{"data":"abc","encoding":"utf8"}`, `{"hex":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}`},
		{"hash_sha256", `{"data":"QQ==","encoding":"base64"}`, `{"hex":"559aead08264d5795d3909718cdd05abd49572e84fe55590eef31a88a08fdffd"}`},
		{"hash_sha512", `{"data":"abc"}`, `{"hex":"ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"}`},
		// A small JSON integer is accepted where a string is expected, and written back as a string.
		{"int_add", `{"a":2,"b":-3}`, `{"value":"-1"}`},
		{"int_add", `{"a":9223372036854775807,"b":"0"}`, `{"value":"9223372036854775807"}`},
		{"int_add", `{"a":-9223372036854775808,"b":"0"}`, `{"value":"-9223372036854775808"}`},
		// The optional version field, and whitespace as llama-server's grammar lets a model write it.
		{"int_add", `{"v":1,"a":"2","b":"3"}`, `{"value":"5"}`},
		{"int_add", "{\"a\"\n   :\"78\", \"b\":\"144431431\"}", `{"value":"144431509"}`},
	} {
		wantResult(t, r, c.op, c.args, c.want)
	}
}

// Identities over random operands up to the size limit, against a value computed another way.
func TestExactOnRandomOperands(t *testing.T) {
	r := registry(t)
	rng := rand.New(rand.NewPCG(1, 2))
	random := func() string {
		digits := 1 + rng.IntN(600)
		var b strings.Builder
		if rng.IntN(2) == 0 {
			b.WriteByte('-')
		}
		b.WriteByte(byte('1' + rng.IntN(9)))
		for range digits - 1 {
			b.WriteByte(byte('0' + rng.IntN(10)))
		}
		return b.String()
	}
	value := func(op, args, field string) string {
		t.Helper()
		resp := run(t, r, op, args)
		if !resp.OK {
			t.Fatalf("%s(%s): %s", op, args, resp.Encode())
		}
		var m map[string]string
		if err := json.Unmarshal(resp.Result, &m); err != nil {
			t.Fatal(err)
		}
		return m[field]
	}
	for range 300 {
		a, b := random(), random()
		ab := fmt.Sprintf(`{"a":%q,"b":%q}`, a, b)
		sum := value("int_add", ab, "value")
		if got := value("int_sub", fmt.Sprintf(`{"a":%q,"b":%q}`, sum, b), "value"); got != a {
			t.Fatalf("(%s + %s) - b = %s", a, b, got)
		}
		product := value("int_mul", ab, "value")
		if got := value("int_div", fmt.Sprintf(`{"a":%q,"b":%q}`, product, b), "quotient"); got != a {
			t.Fatalf("(%s * %s) / b = %s", a, b, got)
		}
		if got := value("int_div", fmt.Sprintf(`{"a":%q,"b":%q}`, product, b), "remainder"); got != "0" {
			t.Fatalf("remainder of a product = %s", got)
		}
		q := value("int_div", ab, "quotient")
		rem := value("int_div", ab, "remainder")
		back := value("int_add", fmt.Sprintf(`{"a":%q,"b":%q}`, value("int_mul", fmt.Sprintf(`{"a":%q,"b":%q}`, q, b), "value"), rem), "value")
		if back != a {
			t.Fatalf("q*b + r = %s, want %s", back, a)
		}
	}
}

func TestMalformedAndInvalidArguments(t *testing.T) {
	r := registry(t)
	for _, c := range []struct {
		op, args string
		code     Code
	}{
		{"int_add", ``, CodeMalformedArguments},
		{"int_add", `{`, CodeMalformedArguments},
		{"int_add", `{"a":"1","b":"2"`, CodeMalformedArguments},
		{"int_add", `nonsense`, CodeMalformedArguments},
		{"int_add", `[1,2]`, CodeMalformedArguments},
		{"int_add", `"a"`, CodeMalformedArguments},
		{"int_add", `5`, CodeMalformedArguments},
		{"int_add", `null`, CodeMalformedArguments},
		{"int_add", `{"a":"1","b":"2"} trailing`, CodeMalformedArguments},
		{"int_add", `{"a":"1","b":"2"}{"a":"1"}`, CodeMalformedArguments},
		{"int_add", `{"a":"1","a":"2","b":"3"}`, CodeMalformedArguments},
		{"int_add", "{\"a\":\"\xff\",\"b\":\"2\"}", CodeMalformedArguments},
		{"int_add", `{"a":[[[[[[[[[[1]]]]]]]]]],"b":"2"}`, CodeMalformedArguments},
		{"array_sum", `{"values":[{"a":1,"a":2}]}`, CodeMalformedArguments},

		{"int_add", `{}`, CodeInvalidArgument},
		{"int_add", `{"a":"1"}`, CodeInvalidArgument},
		{"int_add", `{"a":"1","b":"2","c":"3"}`, CodeInvalidArgument},
		{"int_add", `{"A":"1","b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":null,"b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":true,"b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":["1"],"b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":"abc","b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":"","b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":"-","b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":"+1","b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":"007","b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":"-0","b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":"1.5","b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":"1e3","b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":"0x10","b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":" 1","b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":"1_000","b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":1.5,"b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":1.0,"b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":1e3,"b":"2"}`, CodeInvalidArgument},
		// Past int64 a JSON number would reach the next prompt as another number.
		{"int_add", `{"a":9223372036854775808,"b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":-9223372036854775809,"b":"2"}`, CodeInvalidArgument},
		{"int_add", `{"a":99999999999999999999999,"b":"2"}`, CodeInvalidArgument},
		{"int_pow", `{"base":"2","exponent":"-1"}`, CodeInvalidArgument},
		{"array_sum", `{"values":null}`, CodeInvalidArgument},
		{"array_sum", `{"values":"1"}`, CodeInvalidArgument},
		{"array_sum", `{"values":[null]}`, CodeInvalidArgument},
		{"array_sum", `{"values":["1",2.5]}`, CodeInvalidArgument},
		{"array_min", `{"values":[]}`, CodeInvalidArgument},
		{"array_max", `{"values":[]}`, CodeInvalidArgument},
		{"array_sort", `{"values":["1"],"descending":"yes"}`, CodeInvalidArgument},
		{"hash_sha256", `{}`, CodeInvalidArgument},
		{"hash_sha256", `{"data":null}`, CodeInvalidArgument},
		{"hash_sha256", `{"data":5}`, CodeInvalidArgument},
		{"hash_sha256", `{"data":"zz","encoding":"hex"}`, CodeInvalidArgument},
		{"hash_sha256", `{"data":"***","encoding":"base64"}`, CodeInvalidArgument},
		{"hash_sha256", `{"data":"a","encoding":"rot13"}`, CodeInvalidArgument},
		{"hash_sha256", `{"data":"a","encoding":""}`, CodeInvalidArgument},
		{"hash_sha256", `{"data":"a","encoding":null}`, CodeInvalidArgument},
		{"hash_sha256", `{"data":"QR==","encoding":"base64"}`, CodeInvalidArgument}, // non-zero trailing bits
		{"hash_sha256", `{"data":"Q\nQ==","encoding":"base64"}`, CodeInvalidArgument},
		{"hash_sha256", `{"data":"QQ==\r\n","encoding":"base64"}`, CodeInvalidArgument},

		{"int_add", `{"v":2,"a":"1","b":"2"}`, CodeUnsupportedVersion},
		{"int_add", `{"v":"1","a":"1","b":"2"}`, CodeUnsupportedVersion},
		{"int_div", `{"a":"1","b":"0"}`, CodeDivisionByZero},
		{"int_div", `{"a":"0","b":"0"}`, CodeDivisionByZero},
		{"no_such_op", `{}`, CodeUnknownOp},
		{"", `{}`, CodeUnknownOp},
		{"INT_ADD", `{"a":"1","b":"2"}`, CodeUnknownOp},
	} {
		wantCode(t, r, c.op, c.args, c.code)
	}
}

func TestLimits(t *testing.T) {
	lim := DefaultLimits()
	lim.MaxIntBits = 64
	lim.MaxArrayLen = 3
	lim.MaxArgBytes = 400
	lim.MaxResultBytes = 80
	r, err := NewRegistry(Builtin(), nil, lim)
	if err != nil {
		t.Fatal(err)
	}
	max := "18446744073709551615" // 2^64-1, the largest value of 64 bits
	over := "18446744073709551616"

	wantResult(t, r, "int_add", fmt.Sprintf(`{"a":%q,"b":"0"}`, max), fmt.Sprintf(`{"value":%q}`, max))
	wantCode(t, r, "int_add", fmt.Sprintf(`{"a":%q,"b":"0"}`, over), CodeInputTooLarge)
	wantCode(t, r, "int_add", fmt.Sprintf(`{"a":%q,"b":"1"}`, max), CodeOverflow)
	wantCode(t, r, "int_sub", fmt.Sprintf(`{"a":"-%s","b":"1"}`, max), CodeOverflow)
	wantCode(t, r, "int_mul", `{"a":"4294967296","b":"4294967296"}`, CodeOverflow)
	wantResult(t, r, "int_mul", `{"a":"4294967296","b":"4294967295"}`, `{"value":"18446744069414584320"}`)
	wantResult(t, r, "int_pow", `{"base":"2","exponent":"63"}`, `{"value":"9223372036854775808"}`)
	wantCode(t, r, "int_pow", `{"base":"2","exponent":"64"}`, CodeOverflow)
	wantCode(t, r, "int_pow", `{"base":"3","exponent":"41"}`, CodeOverflow)
	wantResult(t, r, "int_pow", `{"base":"3","exponent":"40"}`, `{"value":"12157665459056928801"}`)
	wantCode(t, r, "int_pow", `{"base":"2","exponent":"18446744073709551615"}`, CodeOverflow)
	wantCode(t, r, "array_product", fmt.Sprintf(`{"values":[%q,"2"]}`, max), CodeOverflow)
	wantCode(t, r, "array_sum", fmt.Sprintf(`{"values":[%q,"1"]}`, max), CodeOverflow)
	wantResult(t, r, "array_sum", `{"values":["1","2","3"]}`, `{"value":"6"}`)
	// The limit holds at every step, not only at the end: it is what bounds a long product's work.
	// So a partial result past it is overflow, even when the final value would fit.
	wantCode(t, r, "array_product", fmt.Sprintf(`{"values":[%q,"2","0"]}`, max), CodeOverflow)
	wantCode(t, r, "array_sum", fmt.Sprintf(`{"values":[%q,"1","-1"]}`, max), CodeOverflow)
	wantCode(t, r, "array_sum", `{"values":["1","2","3","4"]}`, CodeInputTooLarge)
	wantCode(t, r, "hash_sha256", fmt.Sprintf(`{"data":%q}`, strings.Repeat("x", 400)), CodeInputTooLarge)
	wantCode(t, r, "hash_sha512", `{"data":"abc"}`, CodeResultTooLarge) // 128 hex digits

	// The length check has to come before the parse of a very long number, whatever its value.
	wantCode(t, r, "int_add", fmt.Sprintf(`{"a":%q,"b":"0"}`, strings.Repeat("0", 200)), CodeInputTooLarge)

	// The default limit's boundary: 2^4096 - 1 goes in, 2^4096 doesn't, and 2^4095 * 2 overflows.
	d := registry(t)
	two4095 := new(big.Int).Lsh(big.NewInt(1), 4095).String()
	wantResult(t, d, "int_pow", `{"base":"2","exponent":"4095"}`, fmt.Sprintf(`{"value":%q}`, two4095))
	wantCode(t, d, "int_mul", fmt.Sprintf(`{"a":%q,"b":"2"}`, two4095), CodeOverflow)
	wantCode(t, d, "int_pow", `{"base":"2","exponent":"4096"}`, CodeOverflow)
}

func TestAllowlist(t *testing.T) {
	r, err := NewRegistry(Builtin(), []string{"int_add"}, DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	wantResult(t, r, "int_add", `{"a":"1","b":"2"}`, `{"value":"3"}`)
	wantCode(t, r, "int_mul", `{"a":"1","b":"2"}`, CodeOpNotAllowed)
	wantCode(t, r, "no_such_op", `{}`, CodeUnknownOp)
	if tools := r.Tools(); len(tools) != 1 || !strings.Contains(string(tools[0]), `"name":"int_add"`) {
		t.Fatalf("tools %s", tools)
	}

	none, err := NewRegistry(Builtin(), []string{}, DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	wantCode(t, none, "int_add", `{"a":"1","b":"2"}`, CodeOpNotAllowed)
	if len(none.Tools()) != 0 {
		t.Fatal("an empty allowlist advertises tools")
	}

	if _, err := NewRegistry(Builtin(), []string{"int_ad"}, DefaultLimits()); err == nil {
		t.Fatal("an allowlist with a typo passes")
	}
	if _, err := NewRegistry(append(Builtin(), Builtin()[0]), nil, DefaultLimits()); err == nil {
		t.Fatal("two operations of one name pass")
	}
	if _, err := NewRegistry(Builtin(), nil, Limits{}); err == nil {
		t.Fatal("zero limits pass")
	}
}

func TestResponseEnvelope(t *testing.T) {
	r := registry(t)
	if got, want := run(t, r, "int_add", `{"a":"1","b":"2"}`).Encode(),
		`{"v":1,"id":"x","op":"int_add","ok":true,"result":{"value":"3"}}`; got != want {
		t.Errorf("got  %s\nwant %s", got, want)
	}
	if got, want := run(t, r, "int_div", `{"a":"1","b":"0"}`).Encode(),
		`{"v":1,"id":"x","op":"int_div","ok":false,"error":{"code":"division_by_zero","message":"the divisor is 0"}}`; got != want {
		t.Errorf("got  %s\nwant %s", got, want)
	}
	// A name that isn't an operation doesn't get echoed in the op field, and is clipped in the text.
	long := strings.Repeat("x", 5000)
	resp := r.Execute(context.Background(), Request{ID: "x", Op: long, Args: []byte(`{}`)})
	if resp.Op != "" || len(resp.Encode()) > 300 {
		t.Errorf("unknown op: %s", resp.Encode())
	}
}

func TestCancelledContext(t *testing.T) {
	r := registry(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	resp := r.Execute(ctx, Request{ID: "x", Op: "int_add", Args: []byte(`{"a":"1","b":"2"}`)})
	if resp.OK || resp.Error.Code != CodeCancelled {
		t.Fatalf("%s", resp.Encode())
	}
}

func TestPanicIsAnInternalError(t *testing.T) {
	boom := DefineForTest("boom", func(context.Context, json.RawMessage) (json.RawMessage, *Error) { panic("bang") })
	r, err := NewRegistry([]Op{boom}, nil, DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	wantCode(t, r, "boom", `{}`, CodeInternal)
}

// Same request, same bytes, also under concurrency.
func TestReproducible(t *testing.T) {
	r := registry(t)
	cases := []struct{ op, args string }{
		{"int_mul", `{"a":"987654321987654321","b":"123456789123456789"}`},
		{"array_sort", `{"values":["5","-1","5","3"]}`},
		{"hash_sha256", `{"data":"abc"}`},
		{"int_div", `{"a":"1","b":"0"}`},
		{"no_such_op", `{"a":"1"}`},
		{"int_add", `{"a":"x","b":"1"}`},
	}
	for _, c := range cases {
		want := run(t, r, c.op, c.args).Encode()
		done := make(chan string, 16)
		for range cap(done) {
			go func() {
				done <- r.Execute(context.Background(), Request{ID: "x", Op: c.op, Args: []byte(c.args)}).Encode()
			}()
		}
		for range cap(done) {
			if got := <-done; got != want {
				t.Errorf("%s: %s, then %s", c.op, want, got)
			}
		}
	}
}

// Every operation's schemas are valid JSON, name the fields its decoder takes, and have fixed bytes.
func TestSchemas(t *testing.T) {
	r := registry(t)
	first := r.Tools()
	if !slices.IsSortedFunc(r.Allowed(), func(x, y Op) int { return strings.Compare(x.Name, y.Name) }) {
		t.Error("operations aren't sorted")
	}
	for i, tool := range first {
		var parsed struct {
			Type     string `json:"type"`
			Function struct {
				Name        string `json:"name"`
				Description string `json:"description"`
				Parameters  struct {
					Type                 string                     `json:"type"`
					Properties           map[string]json.RawMessage `json:"properties"`
					Required             []string                   `json:"required"`
					AdditionalProperties *bool                      `json:"additionalProperties"`
				} `json:"parameters"`
			} `json:"function"`
		}
		if err := json.Unmarshal(tool, &parsed); err != nil {
			t.Fatalf("%s: %v", tool, err)
		}
		f := parsed.Function
		if parsed.Type != "function" || f.Name == "" || f.Description == "" || f.Parameters.Type != "object" ||
			f.Parameters.AdditionalProperties == nil || *f.Parameters.AdditionalProperties {
			t.Errorf("%s", tool)
		}
		if !json.Valid(r.Allowed()[i].Output()) {
			t.Errorf("%s: output schema %s", f.Name, r.Allowed()[i].Output())
		}
		// Sending only the required fields, with the wrong type, must be refused for the field's
		// type and not as an unknown field: the schema's names are the decoder's names.
		for _, name := range f.Parameters.Required {
			if _, ok := f.Parameters.Properties[name]; !ok {
				t.Errorf("%s: required field %s has no schema", f.Name, name)
			}
		}
		for name := range f.Parameters.Properties {
			args := fmt.Sprintf(`{%q:{}}`, name)
			resp := run(t, r, f.Name, args)
			if resp.OK || strings.Contains(resp.Error.Message, "unknown field") {
				t.Errorf("%s(%s): %s", f.Name, args, resp.Encode())
			}
		}
	}
	if !reflect.DeepEqual(first, r.Tools()) {
		t.Error("the tools change between calls")
	}
}

// The integer pattern in the schema and the decoder agree.
func TestIntSchemaPattern(t *testing.T) {
	if !strings.Contains(intSchema, `^(0|-?[1-9][0-9]*)$`) {
		t.Fatalf("schema %s", intSchema)
	}
	r := registry(t)
	for _, good := range []string{"0", "1", "-1", "10", "-10"} {
		wantResult(t, r, "int_add", fmt.Sprintf(`{"a":%q,"b":"0"}`, good), fmt.Sprintf(`{"value":%q}`, good))
	}
	for _, bad := range []string{"-0", "00", "01", "-01", "", "-"} {
		wantCode(t, r, "int_add", fmt.Sprintf(`{"a":%q,"b":"0"}`, bad), CodeInvalidArgument)
	}
}

// The hash goes over the bytes as given: two ways of writing one number are two inputs.
func TestHashOfADecimalString(t *testing.T) {
	want := sha256.Sum256([]byte("121932631356500531347203169112635269"))
	wantResult(t, registry(t), "hash_sha256", `{"data":"121932631356500531347203169112635269"}`,
		`{"hex":"`+hex.EncodeToString(want[:])+`"}`)
}

// No operation can reach a file, the network, the clock or a random source, directly. Standard
// library packages reach them internally (math/big imports math/rand, for its ProbablyPrime),
// which is why this checks the direct imports of this package and not its closure.
func TestImportsStayPure(t *testing.T) {
	allowed := map[string]bool{
		"bytes": true, "context": true, "crypto/sha256": true, "crypto/sha512": true,
		"encoding/base64": true, "encoding/hex": true, "encoding/json": true, "errors": true,
		"fmt": true, "io": true, "math/big": true, "slices": true, "strconv": true, "strings": true,
		"unicode/utf8": true,
	}
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		f, err := parser.ParseFile(token.NewFileSet(), name, src, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		checked++
		for _, imp := range f.Imports {
			path, _ := strconv.Unquote(imp.Path.Value)
			if !allowed[path] {
				t.Errorf("%s imports %s", name, path)
			}
		}
	}
	if checked == 0 {
		t.Fatal("no files checked")
	}
}

// Nothing a model can write may panic, and the answer doesn't depend on the attempt.
func FuzzExecute(f *testing.F) {
	for _, s := range []string{
		`{"a":"1","b":"2"}`, `{"base":"2","exponent":"4095"}`, `{"values":["1","2"]}`, `{"data":"x","encoding":"hex"}`,
		`{`, `null`, `{"a":1e999,"b":-0}`, `{"a":"\ud800"}`, `{"values":[[[[]]]]}`, `{"v":1}`, "\xff",
	} {
		f.Add(s)
	}
	r, err := NewRegistry(Builtin(), nil, DefaultLimits())
	if err != nil {
		f.Fatal(err)
	}
	f.Fuzz(func(t *testing.T, args string) {
		for _, op := range r.Allowed() {
			one := r.Execute(context.Background(), Request{ID: "x", Op: op.Name, Args: []byte(args)})
			if one.Error != nil && one.Error.Code == CodeInternal {
				t.Fatalf("%s(%q): %s", op.Name, args, one.Encode())
			}
			if one.OK == (one.Error != nil) {
				t.Fatalf("%s(%q): %s", op.Name, args, one.Encode())
			}
			if !json.Valid([]byte(one.Encode())) {
				t.Fatalf("%s(%q): invalid response %s", op.Name, args, one.Encode())
			}
			two := r.Execute(context.Background(), Request{ID: "x", Op: op.Name, Args: []byte(args)})
			if one.Encode() != two.Encode() {
				t.Fatalf("%s(%q): %s, then %s", op.Name, args, one.Encode(), two.Encode())
			}
		}
	})
}
