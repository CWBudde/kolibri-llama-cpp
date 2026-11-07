package main

import (
	"strings"
	"testing"
)

// seq repeats unit n times and appends the first part of unit up to tail tokens.
func seq(unit []int, n, tail int) []int {
	var out []int
	for range n {
		out = append(out, unit...)
	}
	return append(out, unit[:tail]...)
}

func distinct(from, n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = from + i
	}
	return out
}

func concat(parts ...[]int) []int {
	var out []int
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

func unit(n int) []int { return distinct(100, n) }

func TestDetectLoops(t *testing.T) {
	for _, c := range []struct {
		name   string
		tokens []int
		start  int
		period int
	}{
		{"one token from the first", seq([]int{7}, 256, 0), 0, 1},
		{"exactly the shortest region of period 1", concat(distinct(1, 5), seq([]int{7}, 16, 0)), 5, 1},
		{"period 2, as ', dass, dass'", seq([]int{5, 6}, 128, 0), 0, 2},
		{"period 5 after a prefix", concat(distinct(1, 10), seq(unit(5), 40, 0)), 10, 5},
		{"cut inside a unit", concat(distinct(1, 3), seq(unit(5), 4, 2)), 3, 5},
		{"exactly three units of the longest period", seq(unit(64), 3, 0), 0, 64},
		{"three units of period 30", seq(unit(30), 3, 0), 0, 30},
		{"the unit's rotation reaches back", []int{102, 100, 101, 102, 100, 101, 102, 100, 101, 102, 100, 101, 102, 100, 101, 102}, 0, 3},
		{"a glitch, then the loop", concat(seq([]int{7}, 20, 0), []int{9}, seq([]int{7}, 16, 0)), 21, 1},
		{"a periodic unit reports its own period", seq([]int{5, 6, 5, 6}, 20, 0), 0, 2},
		{"the earliest start beats the shorter period", seq(concat(seq([]int{7}, 16, 0), []int{1, 2, 3}), 4, 16), 0, 19},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := FixedRule.Detect(c.tokens, true)
			if !got.Looping || got.Start != c.start || got.Period != c.period {
				t.Fatalf("Detect = %+v; want start %d, period %d", got, c.start, c.period)
			}
		})
	}
}

// capitalList is the shape of the English continuation the docs record: a
// template repeats while its slots change.
func capitalList() []int {
	var out []int
	for i := range 28 {
		out = append(out, 325, 9255, 279, 2000+i, 735, 3000+i, 13)
	}
	return out
}

func TestDetectNearMisses(t *testing.T) {
	for _, c := range []struct {
		name   string
		tokens []int
	}{
		{"nothing generated", nil},
		{"one token", []int{7}},
		{"distinct tokens", distinct(1, 256)},
		{"a template with changing slots", capitalList()},
		{"one token short of the region", concat(distinct(1, 20), seq([]int{7}, 15, 0))},
		{"two units of a long period", seq(unit(40), 2, 0)},
		{"just under three units", seq(unit(30), 2, 29)},
		{"three units of 5 are under the region", seq(unit(5), 3, 0)},
		{"a period past the cap", seq(unit(65), 3, 0)},
		{"a run that ends before the last token", concat(distinct(1, 10), seq([]int{7}, 100, 0), distinct(200, 40))},
		{"a loop broken by the very last token", concat(seq([]int{7}, 100, 0), []int{9})},
		// the shape of "1. Berlin 2. Berlin 3. Berlin": the counter breaks the period
		{"a numbered list of one word", func() []int {
			var out []int
			for i := range 64 {
				out = append(out, 3000+i, 13, 7)
			}
			return out
		}()},
		{"counting with a repeated separator", func() []int {
			var out []int
			for i := range 100 {
				out = append(out, 500+i, 11)
			}
			return out
		}()},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := FixedRule.Detect(c.tokens, true); got.Looping {
				t.Fatalf("Detect = %+v; want no loop", got)
			}
		})
	}
}

// Legitimate periodic output that the limit cuts off is a false positive of the
// rule. It stands, because the rule cannot tell it from a loop by token IDs, and
// docs/loop-baseline.md says so: a Markdown separator such as "|---|---|---" is
// this shape if it tokenizes to a repeating pair.
func TestDetectCountsPeriodicOutputAtTheLimit(t *testing.T) {
	separator := concat(distinct(1, 40), seq([]int{5, 6}, 8, 0))
	if got := FixedRule.Detect(separator, true); !got.Looping || got.Start != 40 || got.Period != 2 {
		t.Fatalf("Detect = %+v", got)
	}
	if got := FixedRule.Detect(separator, false); got.Looping {
		t.Fatalf("Detect after EOS = %+v", got)
	}
}

// The limit censors detection: a loop counts only if it starts by
// n - max(MinRegion, MinRepeats*p).
func TestDetectCensoredByTheLimit(t *testing.T) {
	const n = 256
	for _, c := range []struct{ period, latest int }{
		{1, n - 16},
		{5, n - 16},
		{6, n - 18},
		{30, n - 90},
		{64, n - 192},
	} {
		tail := func(start int) []int {
			return concat(distinct(5000, start), seq(unit(c.period), n/c.period+1, 0))[:n]
		}
		if got := FixedRule.Detect(tail(c.latest), true); !got.Looping || got.Start != c.latest || got.Period != c.period {
			t.Errorf("period %d from %d: Detect = %+v", c.period, c.latest, got)
		}
		if got := FixedRule.Detect(tail(c.latest+1), true); got.Looping {
			t.Errorf("period %d from %d: Detect = %+v; want it censored", c.period, c.latest+1, got)
		}
	}
}

func TestDetectEOSEscapes(t *testing.T) {
	tokens := seq([]int{7}, 200, 0)
	if got := FixedRule.Detect(tokens, false); got.Looping {
		t.Fatalf("a generation the model ended counts as a loop: %+v", got)
	}
}

func TestScoreAnswer(t *testing.T) {
	berlin := &Answer{Accept: []string{"Berlin"}}
	canberra := &Answer{Accept: []string{"Canberra"}, Reject: []string{"Sydney"}}
	twelve := &Answer{Accept: []string{"12", "zwölf"}}
	wall := &Answer{Accept: []string{"1989"}}
	east := &Answer{Accept: []string{"east"}}
	hundred := &Answer{Accept: []string{"100"}}
	water := &Answer{Accept: []string{"H2O", "H₂O", "H_2O", "H_{2}O", `\text{H}_2\text{O}`, `\mathrm{H}_2\mathrm{O}`}}
	for _, c := range []struct {
		name    string
		answer  *Answer
		form    string
		content string
		want    string
	}{
		{"no known answer", nil, formRaw, "whatever", scoreNone},
		{"plain", berlin, formRaw, " Berlin.", scoreCorrect},
		{"case is ignored", berlin, formChat, "BERLIN", scoreCorrect},
		{"inside a longer word", berlin, formRaw, " Berliner Luft", scoreWrong},
		{"a loop without the answer", berlin, formRaw, " Deutschland Deutschland Deutschland", scoreWrong},
		{"nothing generated", berlin, formRaw, "  \n", scoreUnanswered},
		{"accepted before rejected", canberra, formRaw, "Canberra, not Sydney", scoreCorrect},
		{"rejected before accepted", canberra, formRaw, "Sydney, or rather Canberra", scoreWrong},
		{"a number inside a number", twelve, formRaw, " 112 and 120", scoreWrong},
		{"a number", twelve, formRaw, " 12.", scoreCorrect},
		{"umlauts stay one word", twelve, formRaw, " zwölfte", scoreWrong},
		{"a second term", twelve, formRaw, " zwölf", scoreCorrect},
		{"a wrong year, then the right one", wall, formRaw, " 1961. Sie wurde 1961 gebaut und erst 1989 geöffnet.", scoreWrong},
		{"a wrong direction, then the right one", east, formRaw, " west. Not in the east.", scoreWrong},
		{"the first sentence decides", berlin, formRaw, " Berlin. Sie ist groß. Paris auch.", scoreCorrect},
		{"a later line does not count", berlin, formRaw, " Paris\nBerlin", scoreWrong},
		{"leading space is not a sentence", berlin, formRaw, "\n\n Berlin.", scoreCorrect},
		{"a decimal is not a sentence end", hundred, formRaw, " etwa 99.9 oder 100 Grad.", scoreCorrect},
		{"a long sentence is cut at the window", berlin, formRaw, strings.Repeat("x ", answerWindow) + " Berlin", scoreWrong},
		{"the formula in LaTeX", water, formChat, "Die Formel ist $\\text{H}_2\\text{O}$.", scoreCorrect},
		{"the formula as plain LaTeX", water, formChat, "Sie lautet $H_2O$.", scoreCorrect},
		{"the first sentence after reasoning", berlin, formChatThink, "<think>hm</think>\n\nParis. Berlin.", scoreWrong},
		{"reasoning does not count", berlin, formChatThink, "<think>Berlin?</think>\n\nParis", scoreWrong},
		{"the answer after reasoning", berlin, formChatThink, "<think>hm</think>\n\nBerlin", scoreCorrect},
		{"reasoning never closed", berlin, formChatThink, "<think>Berlin Berlin", scoreUnanswered},
		{"nothing after reasoning", berlin, formChatThink, "<think>Berlin</think>", scoreUnanswered},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := ScoreAnswer(c.answer, c.form, c.content); got != c.want {
				t.Fatalf("ScoreAnswer = %q; want %q", got, c.want)
			}
		})
	}
}

// Mentioned is the occurrence rate accuracy is not: it counts what a later
// sentence says, so a longer generation can move it.
func TestAnswerMentioned(t *testing.T) {
	wall := &Answer{Accept: []string{"1989"}}
	canberra := &Answer{Accept: []string{"Canberra"}, Reject: []string{"Sydney"}}
	for _, c := range []struct {
		name    string
		answer  *Answer
		form    string
		content string
		want    bool
	}{
		{"no known answer", nil, formRaw, "1989", false},
		{"after a wrong first answer", wall, formRaw, " 1961. Sie wurde 1961 gebaut und erst 1989 geöffnet.", true},
		{"not at all", wall, formRaw, " 1961.", false},
		{"nothing generated", wall, formRaw, " ", false},
		{"a rejected term first", canberra, formRaw, "Sydney. Canberra.", false},
		{"only in the reasoning", wall, formChatThink, "<think>1989</think>1961", false},
		{"reasoning never closed", wall, formChatThink, "<think>1989", false},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := AnswerMentioned(c.answer, c.form, c.content); got != c.want {
				t.Fatalf("AnswerMentioned = %v; want %v", got, c.want)
			}
		})
	}
}
