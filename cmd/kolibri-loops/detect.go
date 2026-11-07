package main

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Rule is the loop detector's definition. FixedRule is the only one a run or a
// report uses; the other values exist for the negative controls M72 and M73.
type Rule struct {
	// MaxPeriod is the longest repeating unit, in tokens.
	MaxPeriod int `json:"max_period"`
	// MinRepeats is how many whole units the periodic region must hold.
	MinRepeats int `json:"min_repeats"`
	// MinRegion is the shortest periodic region, in tokens.
	MinRegion int `json:"min_region"`
}

// FixedRule was committed before any real run and must not move afterwards
// (docs/loop-baseline.md, "The loop rule").
var FixedRule = Rule{MaxPeriod: 64, MinRepeats: 3, MinRegion: 16}

// Loop is what the detector says about one generation.
type Loop struct {
	Looping bool `json:"looping"`
	// Start is the index in the generated tokens where the periodic region
	// starts, so the time to loop in tokens.
	Start int `json:"start"`
	// Period is the length of the repeating unit, in tokens.
	Period int `json:"period"`
}

// Detect applies the rule to the generated token IDs. stoppedAtLimit says that
// generation ended because of n_predict: a generation the model ended itself
// escaped whatever it repeated before, so it never counts as a loop.
//
// A loop is a region of the generation that holds period p up to its last
// token (g[i] == g[i-p] for every i from start+p on), with at least MinRegion
// tokens and MinRepeats whole units. Among the periods that qualify, the
// earliest start wins and then the shortest period.
func (r Rule) Detect(tokens []int, stoppedAtLimit bool) Loop {
	if !stoppedAtLimit {
		return Loop{}
	}
	n := len(tokens)
	best := Loop{}
	for p := 1; p <= r.MaxPeriod && p < n; p++ {
		i := n - 1
		for i >= p && tokens[i] == tokens[i-p] {
			i--
		}
		start := i - p + 1 // i is the last index that breaks the period, or p-1
		length := n - start
		if length < r.MinRegion || length < r.MinRepeats*p {
			continue
		}
		if !best.Looping || start < best.Start {
			best = Loop{Looping: true, Start: start, Period: p}
		}
	}
	return best
}

// Answer is a corpus item's known answer. Terms match as whole words, ignoring
// case.
type Answer struct {
	Accept []string `json:"accept"`
	Reject []string `json:"reject,omitempty"`
}

// Scores of a generation against a known answer.
const (
	scoreNone       = ""           // the item has no known answer
	scoreCorrect    = "correct"    // in the first sentence, an accepted term comes first
	scoreWrong      = "wrong"      // in the first sentence, a rejected term comes first, or no term at all
	scoreUnanswered = "unanswered" // nothing was generated after the reasoning, or reasoning never closed
)

// thinkEnd closes the reasoning block. llama-server keeps it in content.
const thinkEnd = "</think>"

// answerRegion is the text that can hold an answer: with thinking on, what comes
// after the last closing think tag. It is false if there is no such text.
func answerRegion(form, content string) (string, bool) {
	region := content
	if form == formChatThink {
		i := strings.LastIndex(content, thinkEnd)
		if i < 0 {
			return "", false
		}
		region = content[i+len(thinkEnd):]
	}
	region = strings.TrimSpace(region)
	return region, region != ""
}

// answerWindow caps the first sentence, in runes, for text that never ends one,
// so that a generation's length does not move the score.
const answerWindow = 200

// firstSentence is the text up to the first line break, or the first . ! ? that
// a space or the end follows, at most answerWindow runes.
func firstSentence(text string) string {
	runes := []rune(text)
	for i, r := range runes {
		if i == answerWindow {
			return string(runes[:i])
		}
		if r == '\n' {
			return string(runes[:i])
		}
		if (r == '.' || r == '!' || r == '?') && (i+1 == len(runes) || unicode.IsSpace(runes[i+1])) {
			return string(runes[:i+1])
		}
	}
	return text
}

func scoreTerms(a *Answer, text string) string {
	accept, reject := firstTerm(text, a.Accept), firstTerm(text, a.Reject)
	if accept >= 0 && (reject < 0 || accept < reject) {
		return scoreCorrect
	}
	return scoreWrong
}

// ScoreAnswer scores the first sentence of the answer, so what a generation goes
// on to say later does not count, and neither does how long it runs.
func ScoreAnswer(a *Answer, form, content string) string {
	if a == nil {
		return scoreNone
	}
	region, ok := answerRegion(form, content)
	if !ok {
		return scoreUnanswered
	}
	return scoreTerms(a, firstSentence(region))
}

// AnswerMentioned looks for an accepted term in all of the answer. It is an
// occurrence rate, not an accuracy: an accepted term after a wrong first answer
// counts, and the longer the generation the likelier that is. So rejected terms
// play no part here, only in ScoreAnswer's order.
func AnswerMentioned(a *Answer, form, content string) bool {
	if a == nil {
		return false
	}
	region, ok := answerRegion(form, content)
	return ok && firstTerm(region, a.Accept) >= 0
}

// firstTerm returns the byte offset of the first whole-word match of any term
// in text, or -1.
func firstTerm(text string, terms []string) int {
	lower := strings.ToLower(text)
	best := -1
	for _, t := range terms {
		t = strings.ToLower(t)
		for from := 0; from <= len(lower); {
			i := strings.Index(lower[from:], t)
			if i < 0 {
				break
			}
			i += from
			if wholeWord(lower, i, i+len(t)) {
				if best < 0 || i < best {
					best = i
				}
				break
			}
			from = i + 1
		}
	}
	return best
}

func wholeWord(s string, start, end int) bool {
	wordRune := func(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) }
	if before, n := utf8.DecodeLastRuneInString(s[:start]); n > 0 && wordRune(before) {
		return false
	}
	if after, n := utf8.DecodeRuneInString(s[end:]); n > 0 && wordRune(after) {
		return false
	}
	return true
}
