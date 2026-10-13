package locality

import (
	"encoding/json"
	"fmt"
	"io"
	"slices"
)

// EOS are the end-of-generation tokens <|endoftext|> and <|im_end|>, which tools/locality/workloads.py
// strips from a generated answer.
var EOS = []int32{127901, 127906}

// Answer is a generated answer of a workload trace (testdata/locality/<workload>.json): the
// length of the prompt it was generated from, its generated token IDs without EOS and whether
// generation stopped at an EOS or at the token limit.
type Answer struct {
	Prompt int
	Tokens []int32
	EOS    bool
}

// Decoded is the number of the answer's tokens a server passed through the model while
// generating: every generated token is fed back to sample the next one, except the last when
// generation stopped at the limit. That token is first processed in the next prompt's prefill.
func (a Answer) Decoded() int {
	if a.EOS || len(a.Tokens) == 0 {
		return len(a.Tokens)
	}
	return len(a.Tokens) - 1
}

// ReadAnswers reads a trace's generated answers in conversation order.
func ReadAnswers(r io.Reader) ([]Answer, error) {
	var trace struct {
		Messages []struct {
			Answer *struct {
				NPrompt *int    `json:"n_prompt"`
				Tokens  []int32 `json:"tokens"`
				Stop    string  `json:"stop"`
			} `json:"answer"`
		} `json:"messages"`
	}
	if err := json.NewDecoder(r).Decode(&trace); err != nil {
		return nil, err
	}
	var out []Answer
	for i, m := range trace.Messages {
		a := m.Answer
		if a == nil {
			continue
		}
		if a.NPrompt == nil || (a.Stop != "eos" && a.Stop != "limit") {
			return nil, fmt.Errorf("message %d: answer without n_prompt or with stop %q, want eos or limit", i, a.Stop)
		}
		var gen []int32
		for _, t := range a.Tokens {
			if !slices.Contains(EOS, t) {
				gen = append(gen, t)
			}
		}
		out = append(out, Answer{Prompt: *a.NPrompt, Tokens: gen, EOS: a.Stop == "eos"})
	}
	return out, nil
}

// CheckAnswers checks that the token IDs of the whole conversation hold every answer's generated
// tokens right after its prompt.
func CheckAnswers(ids []int32, answers []Answer) error {
	for i, a := range answers {
		end := a.Prompt + len(a.Tokens)
		if a.Prompt < 0 || end > len(ids) || !slices.Equal(ids[a.Prompt:end], a.Tokens) {
			return fmt.Errorf("answer %d: the token IDs do not hold its %d generated tokens at %d", i, len(a.Tokens),
				a.Prompt)
		}
	}
	return nil
}

// OnlineSpans is the order in which a server with a prompt cache meets a conversation of tokens
// tokens: for each answer it prefills the prompt's tokens it has not processed yet in ubatches of
// ubatch tokens, then decodes the answer token by token (Decoded of them); the tokens after the
// last answer are a final prefill.
func OnlineSpans(tokens int, answers []Answer, ubatch int) ([]Span, error) {
	if ubatch < 1 {
		return nil, fmt.Errorf("ubatch %d, want at least 1", ubatch)
	}
	var out []Span
	prefill := func(first, end int) {
		for t := first; t < end; t += ubatch {
			out = append(out, Span{Prefill, t, min(t+ubatch, end)})
		}
	}
	done, end := 0, 0 // the tokens processed; where the last answer's generated tokens end
	for i, a := range answers {
		if a.Prompt < end || a.Prompt+len(a.Tokens) > tokens {
			return nil, fmt.Errorf("answer %d: prompt of %d tokens and %d generated, after an answer ending at %d, "+
				"of %d tokens", i, a.Prompt, len(a.Tokens), end, tokens)
		}
		prefill(done, a.Prompt)
		for t := a.Prompt; t < a.Prompt+a.Decoded(); t++ {
			out = append(out, Span{Decode, t, t + 1})
		}
		done = a.Prompt + a.Decoded()
		end = a.Prompt + len(a.Tokens)
	}
	prefill(done, tokens)
	return out, nil
}
