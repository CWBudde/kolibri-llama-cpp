package locality

import (
	"strings"
	"testing"
)

func TestSplitAfter(t *testing.T) {
	answers, _, err := ReadAnswers(strings.NewReader(traceJSON))
	if err != nil {
		t.Fatal(err)
	}
	// TestOnlineSpans' spans: answer 0 is decoded at 3 and 4, then the prefill from 5 on is the
	// second request's
	spans, err := OnlineSpans(13, answers, false, 2)
	if err != nil {
		t.Fatal(err)
	}
	cut, err := SplitAfter(spans, answers, 1)
	if err != nil || cut != 4 || spans[cut].First != 5 || spans[cut].Phase != Prefill {
		t.Fatalf("SplitAfter = %d, %v", cut, err)
	}
	for _, k := range []int{0, 2, -1} {
		if _, err := SplitAfter(spans, answers, k); err == nil {
			t.Errorf("a split after %d of %d answers accepted", k, len(answers))
		}
	}
	// spans that do not hold the answer's boundary
	if _, err := SplitAfter(TraceOrder(4), answers, 1); err == nil {
		t.Error("spans ending before the boundary accepted")
	}
	if _, err := SplitAfter([]Span{{Prefill, 0, 13}}, answers, 1); err == nil {
		t.Error("a span across the boundary accepted")
	}
}
