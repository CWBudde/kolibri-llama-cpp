package main

import (
	"bufio"
	"encoding/json"
	"os"
	"slices"
	"testing"
)

func TestAssignRuntimeIDs(t *testing.T) {
	vocab := map[string]int{"a": 0, "b": 1, "c": 2}
	added := []AddedToken{
		{ID: 3, Content: "<x>"},
		{ID: 7, Content: "<y>"}, // the file skips 4..6
		{ID: 1, Content: "b"},   // already in the vocab: keeps its ID
		{ID: 9, Content: "<z>"},
	}
	assignRuntimeIDs(vocab, added)
	var got []int
	for _, a := range added {
		got = append(got, a.RuntimeID)
	}
	if want := []int{3, 4, 1, 5}; !slices.Equal(got, want) {
		t.Errorf("runtime IDs %v, want %v", got, want)
	}
	if gaps := idGaps(vocab, added, func(a AddedToken) int { return a.ID }, -1); !slices.Equal(gaps, []int{4, 5, 6, 8}) {
		t.Errorf("file gaps %v", gaps)
	}
	if gaps := idGaps(vocab, added, func(a AddedToken) int { return a.RuntimeID }, 7); !slices.Equal(gaps, []int{6, 7}) {
		t.Errorf("runtime gaps %v", gaps)
	}
}

// TestRuntimeIDsMatchGolden cross-checks the committed inventory against the
// tokenizer golden file, which tools/tokenizer/golden.py generates from the
// real tokenizers library.
func TestRuntimeIDsMatchGolden(t *testing.T) {
	raw, err := os.ReadFile("../../inventory/bf16/summary.json")
	if err != nil {
		t.Fatal(err)
	}
	var sum Summary
	if err := json.Unmarshal(raw, &sum); err != nil {
		t.Fatal(err)
	}
	tok := sum.Tokenizer
	if !slices.Equal(tok.RuntimeUnassigned, []int{127998, 127999}) {
		t.Errorf("runtime_unassigned = %v", tok.RuntimeUnassigned)
	}
	rt := map[string]int{}
	for _, a := range tok.AddedTokens {
		rt[a.Content] = a.RuntimeID
	}

	f, err := os.Open("../../testdata/tokenizer/golden.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(nil, 1<<20)
	want := map[string][]int{
		"special/reserved-shifted": {rt["<|reserved-token-2|>"], rt["<|reserved-token-3|>"], rt["<|reserved-token-76|>"]},
		"special/role-tokens":      {rt["<|text|>"], rt["<|endoftext|>"], rt["<|pad|>"], rt["<|chat|>"], rt["<|/role|>"]},
		"special/adjacent":         {rt["<|im_end|>"], rt["<|im_start|>"], rt["<|im_end|>"]},
	}
	for sc.Scan() {
		var c struct {
			Name string `json:"name"`
			IDs  []int  `json:"ids"`
		}
		if err := json.Unmarshal(sc.Bytes(), &c); err != nil {
			t.Fatal(err)
		}
		if w, ok := want[c.Name]; ok {
			if !slices.Equal(c.IDs, w) {
				t.Errorf("%s: golden %v, inventory runtime IDs %v", c.Name, c.IDs, w)
			}
			delete(want, c.Name)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	for name := range want {
		t.Errorf("%s missing from golden file", name)
	}
}
