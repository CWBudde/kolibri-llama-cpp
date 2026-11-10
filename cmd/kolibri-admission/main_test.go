package main

import (
	"bytes"
	"math/rand"
	"slices"
	"strings"
	"testing"

	"kolibri-llm/internal/locality"
)

// conversation is a random trace of two prompts and answers over 2 layers, 3 experts per token.
func conversation() (locality.Selections, []locality.Span, int) {
	rng := rand.New(rand.NewSource(11))
	const layers, k, nExpert = 2, 3, 48
	sel := locality.Selections{Layers: layers, K: k}
	var spans []locality.Span
	cut := 0
	for i, n := range [][2]int{{40, 30}, {25, 90}} {
		if i == 1 {
			cut = len(spans)
		}
		spans = append(spans, locality.Span{Phase: locality.Prefill, First: sel.Tokens, End: sel.Tokens + n[0]})
		for t := sel.Tokens + n[0]; t < sel.Tokens+n[0]+n[1]; t++ {
			spans = append(spans, locality.Span{Phase: locality.Decode, First: t, End: t + 1})
		}
		sel.Tokens += n[0] + n[1]
	}
	for range layers * sel.Tokens {
		for _, e := range rng.Perm(nExpert)[:k] {
			sel.IDs = append(sel.IDs, int32(e))
		}
	}
	return sel, spans, cut
}

func TestEvaluate(t *testing.T) {
	sel, spans, cut := conversation()
	a := Assumptions{Ubatch: 512, Batch: 2048, Batches: "server", Waves: true, LastLayer: "outputs", First: []int{16, 64}}
	rows, err := evaluate("m", "w", sel, spans, cut, 48, []int{12, 24}, a)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2*len(locality.AdmissionVariants) {
		t.Fatalf("%d rows, want %d", len(rows), 2*len(locality.AdmissionVariants))
	}
	for i, r := range rows {
		if want := []int{12, 24}[i/5]; r.Slots != want {
			t.Errorf("row %d: %d slots, want %d", i, r.Slots, want)
		}
		if want := locality.AdmissionVariants[i%5].String(); r.Variant != want {
			t.Errorf("row %d: variant %s, want %s", i, r.Variant, want)
		}
		// the second answer decodes 90 tokens of 2 layers of 3 experts
		if r.All.Tokens != 90 || r.All.Needed != 90*2*3 || r.First[0].Tokens != 16 || r.Steady.Tokens != 26 {
			t.Errorf("row %d: %+v", i, r.AdmissionResult)
		}
		if r.Prompts != 1 || r.Layers != 2 {
			t.Errorf("row %d: %d prompts, %d layers", i, r.Prompts, r.Layers)
		}
	}
	again, err := evaluate("m", "w", sel, spans, cut, 48, []int{12, 24}, a)
	if err != nil {
		t.Fatal(err)
	}
	var one, two bytes.Buffer
	writeMarkdown(&one, "cmd", a, rows)
	writeMarkdown(&two, "cmd", a, again)
	if one.String() != two.String() {
		t.Error("two runs give different reports")
	}
	md := one.String()
	for _, want := range []string{"| w | today |", "| w | future-lfu (not implementable) |", "first 16", "tokens 65+", "12 slots", "24 slots"} {
		if !strings.Contains(md, want) {
			t.Errorf("the report lacks %q:\n%s", want, md)
		}
	}
}

func TestTotals(t *testing.T) {
	w := func(tokens, needed, hits int) locality.AdmissionWindow {
		return locality.AdmissionWindow{Tokens: tokens, Needed: needed, Hits: hits}
	}
	row := func(wl, v string, all locality.AdmissionWindow, loads int) Row {
		return Row{Model: "m", Workload: wl, Slots: 8, Variant: v, Layers: 2, AdmissionResult: locality.AdmissionResult{
			First: []locality.AdmissionWindow{all}, All: all, Prompts: 1, SeedLoads: loads}}
	}
	rows := []Row{row("a", "today", w(10, 60, 30), 0), row("a", "lfu", w(10, 60, 40), 4),
		row("b", "today", w(5, 30, 10), 0), row("b", "lfu", w(5, 30, 20), 6)}
	got := totals(rows)
	if len(got) != 2 || got[0].Variant != "today" || got[1].Variant != "lfu" || got[1].Workload != "all" {
		t.Fatalf("totals %+v", got)
	}
	if got[1].All != w(15, 90, 60) || got[1].First[0] != w(15, 90, 60) || got[1].SeedLoads != 10 || got[1].Prompts != 2 {
		t.Errorf("lfu total %+v", got[1].AdmissionResult)
	}
	if rows[0].First[0] != w(10, 60, 30) {
		t.Error("totals changed its rows")
	}
}

func TestSplit(t *testing.T) {
	answers := []locality.Answer{{Prompt: 4, Tokens: []int32{1, 2}, EOS: true}, {Prompt: 8, Tokens: []int32{3, 4}, EOS: true}}
	spans, err := locality.OnlineSpans(10, answers, true, 512)
	if err != nil {
		t.Fatal(err)
	}
	cut, err := split(SplitWorkload{Answers: 2, TrainAnswers: 1, FirstTestToken: 6}, spans, answers)
	if err != nil || spans[cut].First != 6 || spans[cut].Phase != locality.Prefill {
		t.Errorf("cut %d, %v", cut, err)
	}
	for _, d := range []SplitWorkload{{2, 1, 5}, {3, 1, 6}, {2, 2, 6}} {
		if _, err := split(d, spans, answers); err == nil {
			t.Errorf("%+v: no error", d)
		}
	}
}

func TestInts(t *testing.T) {
	got, err := ints("64,128")
	if err != nil || !slices.Equal(got, []int{64, 128}) {
		t.Errorf("%v, %v", got, err)
	}
	for _, s := range []string{"", "a", "0", "-3", "8,,9"} {
		if _, err := ints(s); err == nil {
			t.Errorf("%q: no error", s)
		}
	}
}

// TestValidate: the validation replays a prompt in the server's ubatches from empty slots, and
// its counters add up.
func TestValidate(t *testing.T) {
	sel, _, _ := conversation()
	a := Assumptions{Ubatch: 16, Batch: 64, Batches: "server", Waves: true, LastLayer: "outputs", First: []int{16}}
	v, err := validate(sel, 40, 10, locality.Selections{}, 48, 12, a)
	if err != nil {
		t.Fatal(err)
	}
	// 40 tokens: batches end at 20 and 36; the first is 16+4
	var sizes []int
	for _, s := range v.Spans {
		sizes = append(sizes, s.End-s.First)
	}
	if !slices.Equal(sizes, []int{16, 4, 16, 4}) {
		t.Errorf("ubatches %v, want 16 4 16 4", sizes)
	}
	// two layers: the first runs the 4 ubatches, the last the prompt's last token only
	c := v.Replay
	if c.Steps != 5 || c.Needed != c.Hits+c.Misses || c.Evictions > c.Misses || c.Misses-c.Evictions > 2*12 ||
		c.WavesBuilt == 0 || c.Waves > c.WavesBuilt {
		t.Errorf("prefill %+v", c)
	}
	if d := v.Decode; d.Steps != 20 || d.Needed != 10*2*3 || d.Needed != d.Hits+d.Misses || d.WavesBuilt != 0 {
		t.Errorf("decode %+v", d)
	}
	// two startup decodes of the trace's first two tokens: 2 steps each, the second all hits
	w, err := validate(sel, 40, 0, standIn(sel, 2), 48, 12, a)
	if err != nil {
		t.Fatal(err)
	}
	one, err := validate(sel, 40, 0, standIn(sel, 1), 48, 12, a)
	if err != nil {
		t.Fatal(err)
	}
	if w.Replay.Steps != 9 || w.Startup != 2 || w.Replay.Misses != one.Replay.Misses ||
		w.Replay.Needed-one.Replay.Needed != w.Replay.Hits-one.Replay.Hits {
		t.Errorf("with startup decodes %+v, with one %+v", w.Replay, one.Replay)
	}
	var out bytes.Buffer
	writeValidation(&out, 12, []Validation{w})
	if !strings.Contains(out.String(), "(2)+(2)+16+4+16+4") {
		t.Errorf("the table lacks the ubatches:\n%s", out.String())
	}
	if _, err := validate(sel, 40, 1000, locality.Selections{}, 48, 12, a); err == nil {
		t.Error("more decoded tokens than the trace holds: no error")
	}
}
