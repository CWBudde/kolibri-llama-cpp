package locality

// Train/test splits (PLAN Phase 10, Stage A): a trace is split chronologically after one of its
// answers. The training prefix is everything an online server meets up to that answer's decoded
// tokens; the test suffix starts with the next request's prefill. Anything fitted or tuned sees
// only training prefixes; a test suffix is replayed after its prefix, with the state the prefix's
// causal replay left behind (ReplaySegments), and only its counts are scored.

import "fmt"

// SplitAfter is the index of the first test span when the first k answers are the training
// prefix: the span that starts right after answer k-1's decoded tokens. Both sides keep at least
// one answer, so 0 < k < len(answers).
func SplitAfter(spans []Span, answers []Answer, k int) (int, error) {
	if k <= 0 || k >= len(answers) {
		return 0, fmt.Errorf("split: after %d of %d answers, want both sides to keep one", k, len(answers))
	}
	a := answers[k-1]
	at := a.Prompt + a.Decoded()
	for i, s := range spans {
		if s.First == at {
			return i, nil
		}
	}
	return 0, fmt.Errorf("split: no span starts at token %d, after answer %d", at, k-1)
}
