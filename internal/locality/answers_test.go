package locality

import (
	"reflect"
	"strings"
	"testing"
)

const traceJSON = `{"name": "t", "messages": [
 {"role": "system", "content": "s"},
 {"role": "user", "content": "u"},
 {"role": "assistant", "content": "a", "answer": {"n_prompt": 3, "tokens": [10, 11, 127906], "stop": "eos"}},
 {"role": "user", "content": "v"},
 {"role": "assistant", "content": "b", "answer": {"n_prompt": 8, "tokens": [20, 21, 22], "stop": "limit"}},
 {"role": "user", "content": "w"}
]}`

func TestReadAnswers(t *testing.T) {
	got, ended, err := ReadAnswers(strings.NewReader(traceJSON))
	if err != nil {
		t.Fatal(err)
	}
	if ended {
		t.Error("a trace with a user message after its last answer ended with it")
	}
	if _, ended, _ := ReadAnswers(strings.NewReader(`{"messages": [{"role": "user"},
		{"role": "assistant", "answer": {"n_prompt": 1, "tokens": [1], "stop": "eos"}}]}`)); !ended {
		t.Error("a trace ending with an answer did not end with it")
	}
	want := []Answer{{3, []int32{10, 11}, true}, {8, []int32{20, 21, 22}, false}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ReadAnswers = %v, want %v", got, want)
	}
	// at an EOS every generated token was fed back; at the limit the last one was not
	if got[0].Decoded() != 2 || got[1].Decoded() != 2 || (Answer{}).Decoded() != 0 {
		t.Fatalf("Decoded = %d, %d", got[0].Decoded(), got[1].Decoded())
	}
	for _, bad := range []string{
		`{"messages": [{"answer": {"tokens": [1], "stop": "eos"}}]}`,
		`{"messages": [{"answer": {"n_prompt": 1, "tokens": [1], "stop": "word"}}]}`,
		`{"messages": [`,
	} {
		if _, _, err := ReadAnswers(strings.NewReader(bad)); err == nil {
			t.Errorf("%s accepted", bad)
		}
	}
}

func TestCheckAnswers(t *testing.T) {
	answers, _, err := ReadAnswers(strings.NewReader(traceJSON))
	if err != nil {
		t.Fatal(err)
	}
	ids := []int32{1, 2, 3, 10, 11, 127906, 4, 5, 20, 21, 22, 127906, 6}
	if err := CheckAnswers(ids, answers); err != nil {
		t.Fatal(err)
	}
	ids[9] = 0
	if err := CheckAnswers(ids, answers); err == nil {
		t.Error("a changed answer token accepted")
	}
	if err := CheckAnswers(ids[:9], answers); err == nil {
		t.Error("an answer beyond the token IDs accepted")
	}
}

func TestOnlineSpans(t *testing.T) {
	answers, _, err := ReadAnswers(strings.NewReader(traceJSON))
	if err != nil {
		t.Fatal(err)
	}
	// 13 tokens, ubatch 2: prefill [0,3), decode 3 and 4 (the EOS at 5 is not decoded), prefill
	// [5,8), decode 8 and 9 (10, at the limit, is not), prefill [10,13)
	got, err := OnlineSpans(13, answers, false, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []Span{
		{Prefill, 0, 2}, {Prefill, 2, 3}, {Decode, 3, 4}, {Decode, 4, 5},
		{Prefill, 5, 7}, {Prefill, 7, 8}, {Decode, 8, 9}, {Decode, 9, 10},
		{Prefill, 10, 12}, {Prefill, 12, 13},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("OnlineSpans = %v\nwant          %v", got, want)
	}
	// had the conversation ended with the second answer, nothing would send tokens 10-12
	if got, _ := OnlineSpans(13, answers, true, 2); !reflect.DeepEqual(got, want[:8]) {
		t.Fatalf("ended: OnlineSpans = %v\nwant                 %v", got, want[:8])
	}
	if spans, _ := OnlineSpans(4, nil, true, 512); !reflect.DeepEqual(spans, []Span{{Prefill, 0, 4}}) {
		t.Fatalf("no answers: %v", spans)
	}
	for _, bad := range [][]Answer{
		{{Prompt: 5, Tokens: make([]int32, 3)}, {Prompt: 7, Tokens: []int32{1}}}, // inside the first answer
		{{Prompt: 12, Tokens: make([]int32, 3)}},                                 // beyond the tokens
	} {
		if _, err := OnlineSpans(13, bad, false, 2); err == nil {
			t.Errorf("answers %v accepted", bad)
		}
	}
	if _, err := OnlineSpans(13, answers, false, 0); err == nil {
		t.Error("ubatch 0 accepted")
	}
}
