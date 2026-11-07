package main

import (
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
)

// notDone lists what the issue asks for and this tool does not measure, with
// the reason, in every report.
var notDone = []string{
	"Reference logits: llama-server reports the model under test, not the reference. The reference is tools/ref/kolibri_ref.py or vLLM on BF16; records keep this run's top log probabilities for a later comparison.",
	"BF16: issue #66 compares against BF16 before quantizing. A run on another weight, named above, does not replace that run.",
	"Tool use: the issue asks for it at validation, step 4. The corpus has reasoning but no tools.",
}

// RepetitionCell is one (split, form, lang) of the repetition table.
type RepetitionCell struct {
	Split string  `json:"split"`
	Form  string  `json:"form"`
	Lang  string  `json:"lang"`
	N     int     `json:"n"`
	Loops int     `json:"loops"`
	Rate  float64 `json:"rate"`
	// MedianStart and MaxStart are null in a cell with no loop.
	MedianStart *float64 `json:"median_start"`
	MaxStart    *int     `json:"max_start"`
	Periods     []int    `json:"periods"`
	EOS         int      `json:"eos"`
	// Censored counts generations the limit stopped with no detectable loop. What
	// they would have done next is unknown (docs/loop-baseline.md, "What the token
	// limit hides").
	Censored int `json:"censored"`
}

// AnswerCell is one (split, form, lang) of the accuracy table.
type AnswerCell struct {
	Split      string  `json:"split"`
	Form       string  `json:"form"`
	Lang       string  `json:"lang"`
	Scored     int     `json:"scored"`
	Correct    int     `json:"correct"`
	Wrong      int     `json:"wrong"`
	Unanswered int     `json:"unanswered"`
	Accuracy   float64 `json:"accuracy"`
	// Mentioned counts accepted terms anywhere in the answer, not an accuracy.
	Mentioned   int     `json:"mentioned"`
	MentionRate float64 `json:"mention_rate"`
}

// LossCell is one (split, lang) of the held-out loss table. PPL and MeanNLL are
// nil if a target had probability 0, which has no finite loss.
type LossCell struct {
	Split     string   `json:"split"`
	Lang      string   `json:"lang"`
	Texts     int      `json:"texts"`
	Positions int      `json:"positions"`
	Zero      int      `json:"zero_probability"`
	MeanNLL   *float64 `json:"mean_nll"`
	PPL       *float64 `json:"ppl"`
}

// ControlRow compares one documented prompt with this run.
type ControlRow struct {
	ID         string `json:"id"`
	Split      string `json:"split"`
	Documented string `json:"documented"`
	Looping    bool   `json:"looping"`
	// Start and Period are null without a loop: start 0 means the first token.
	Start  *int `json:"start"`
	Period *int `json:"period"`
	// PFirst is the probability of the first token of the loop, PLoop the median
	// probability of the generated tokens after the first unit.
	PFirst *float64 `json:"p_first"`
	PLoop  *float64 `json:"p_loop"`
	Agrees bool     `json:"agrees"`
}

// PerRecord is the detector's result for one record.
type PerRecord struct {
	ID      string `json:"id"`
	Looping bool   `json:"looping"`
	// Start and Period are null without a loop: start 0 means the first token.
	Start     *int   `json:"start"`
	Period    *int   `json:"period"`
	Tokens    int    `json:"tokens"`
	Score     string `json:"score,omitempty"`
	Mentioned bool   `json:"mentioned,omitempty"`
}

// Report is computed from the records alone.
type Report struct {
	Records    int              `json:"records"`
	Items      int              `json:"items"`
	Pins       Pins             `json:"pins"`
	Repetition []RepetitionCell `json:"repetition"`
	Answers    []AnswerCell     `json:"answers"`
	Loss       []LossCell       `json:"loss"`
	Controls   []ControlRow     `json:"controls"`
	PerRecord  []PerRecord      `json:"per_record"`
	// TokensNoProb counts generated tokens the server listed no probability for.
	TokensNoProb int `json:"tokens_without_probability"`
	// LossBans counts the bans measureLoss needed.
	LossBans int      `json:"loss_bans"`
	NotDone  []string `json:"not_done"`
}

func readRecords(out string) ([]Record, error) {
	entries, err := os.ReadDir(recordsDir(out))
	if err != nil {
		return nil, err
	}
	var rs []Record
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".json") || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		r, err := readRecord(filepath.Join(recordsDir(out), e.Name()))
		if err != nil {
			return nil, err
		}
		if e.Name() != r.ID+".json" {
			return nil, fmt.Errorf("%s holds record %s", e.Name(), r.ID)
		}
		rs = append(rs, r)
	}
	return rs, nil
}

var formOrder = []string{formRaw, formChat, formChatThink, formText}

func rank(list []string, s string) int { return slices.Index(list, s) }

// cellLess orders the tables: tuning first, then raw, chat, thinking, then de.
func cellLess(split1, form1, lang1, split2, form2, lang2 string) bool {
	if split1 != split2 {
		return rank([]string{splitTuning, splitHeldOut}, split1) < rank([]string{splitTuning, splitHeldOut}, split2)
	}
	if form1 != form2 {
		return rank(formOrder, form1) < rank(formOrder, form2)
	}
	return lang1 < lang2
}

func medianInts(v []int) float64 {
	s := slices.Sorted(slices.Values(v))
	if len(s)%2 == 1 {
		return float64(s[len(s)/2])
	}
	return (float64(s[len(s)/2-1]) + float64(s[len(s)/2])) / 2
}

func medianFloats(v []float64) float64 {
	s := slices.Sorted(slices.Values(v))
	if len(s)%2 == 1 {
		return s[len(s)/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}

// checkRecord re-applies the checks of a run to what the record stores, so a
// truncated record, or a hand edit these can see, does not reach the tables. An
// edit that stays plausible gets through: records carry no checksum.
func checkRecord(r Record) error {
	if r.Item.ID != r.ID {
		return fmt.Errorf("record %s holds item %s", r.ID, r.Item.ID)
	}
	var req completionRequest
	if err := json.Unmarshal(r.Request, &req); err != nil {
		return fmt.Errorf("record %s: request: %v", r.ID, err)
	}
	s := r.Pins.Settings
	if r.Item.Form == formText {
		if len(r.PromptTokens) < 2 || len(r.Positions) != len(r.PromptTokens)-1 {
			return fmt.Errorf("record %s: %d positions for %d tokens", r.ID, len(r.Positions), len(r.PromptTokens))
		}
		if !slices.Equal(req.Prompt, r.PromptTokens[:len(r.PromptTokens)-1]) || req.NPredict != 1 || req.NProbs != lossNProbs || req.Seed != s.Seed {
			return fmt.Errorf("record %s: the request is not the last position of the pinned settings", r.ID)
		}
		resp := completionResponse{Truncated: r.Truncated, TokensEvaluated: r.TokensEvaluated, GenerationSettings: r.ResolvedSettings}
		if err := checkResolved(req, resp); err != nil {
			return fmt.Errorf("record %s: %v", r.ID, err)
		}
		for i, p := range r.Positions {
			if p.Target != r.PromptTokens[i+1] {
				return fmt.Errorf("record %s: position %d targets %d, the text has %d", r.ID, i+1, p.Target, r.PromptTokens[i+1])
			}
			if err := checkLogprob(p.Logprob); err != nil {
				return fmt.Errorf("record %s: position %d: %v", r.ID, i+1, err)
			}
			if p.Candidates != r.Pins.NVocab {
				return fmt.Errorf("record %s: position %d: %d candidates, the vocabulary has %d", r.ID, i+1, p.Candidates, r.Pins.NVocab)
			}
		}
		return nil
	}
	if !slices.Equal(req.Prompt, r.PromptTokens) {
		return fmt.Errorf("record %s: the request's prompt is not prompt_tokens", r.ID)
	}
	n := s.NPredict
	if r.Item.Form == formChatThink {
		n = s.NPredictThink
	}
	if req.NPredict != n || req.Seed != s.Seed || req.NProbs != s.NProbs {
		return fmt.Errorf("record %s: the request is not the pinned settings", r.ID)
	}
	if len(r.Probs) != len(r.GeneratedTokens) {
		return fmt.Errorf("record %s: %d probabilities for %d tokens", r.ID, len(r.Probs), len(r.GeneratedTokens))
	}
	var listed []tokenProb
	for i, p := range r.Probs {
		if p == nil {
			continue
		}
		if p.ID != r.GeneratedTokens[i] {
			return fmt.Errorf("record %s: token %d has probabilities for %d", r.ID, i, p.ID)
		}
		listed = append(listed, *p)
	}
	resp := completionResponse{Tokens: r.GeneratedTokens, StopType: r.StopType, Truncated: r.Truncated, TokensEvaluated: r.TokensEvaluated, TokensPredicted: r.TokensPredicted, Probs: listed, GenerationSettings: r.ResolvedSettings}
	if _, err := checkCompletion(req, resp); err != nil {
		return fmt.Errorf("record %s: %v", r.ID, err)
	}
	return nil
}

// buildReport computes everything from the records. It refuses records of
// different runs.
func buildReport(rs []Record) (*Report, error) {
	if len(rs) == 0 {
		return nil, fmt.Errorf("no records")
	}
	rep := &Report{Records: len(rs), Items: rs[0].Pins.CorpusItems, Pins: rs[0].Pins, NotDone: notDone}
	// The numbers below use FixedRule and the header prints the pinned rule, so the
	// two must be one. A rule that moved after these records were made is refused,
	// not applied under their label.
	if rule := rep.Pins.Settings.Rule; rule != FixedRule {
		return nil, fmt.Errorf("the records were made with loop rule %+v, this tool's is %+v", rule, FixedRule)
	}
	type key struct{ split, form, lang string }
	rep_ := map[key]*RepetitionCell{}
	ans := map[key]*AnswerCell{}
	starts := map[key][]int{}
	type lossKey struct{ split, lang string }
	loss := map[lossKey]*LossCell{}
	sums := map[lossKey]float64{}
	sort.Slice(rs, func(i, j int) bool { return rs[i].ID < rs[j].ID })
	for _, r := range rs {
		if diff, err := diffPins(r.Pins, rep.Pins); err != nil {
			return nil, err
		} else if len(diff) > 0 {
			return nil, fmt.Errorf("records %s and %s are of different runs: %s", rs[0].ID, r.ID, strings.Join(diff, ", "))
		}
		if err := checkRecord(r); err != nil {
			return nil, err
		}
		it := r.Item
		if it.Form == formText {
			k := lossKey{it.Split, it.Lang}
			c := loss[k]
			if c == nil {
				c = &LossCell{Split: it.Split, Lang: it.Lang}
				loss[k] = c
			}
			c.Texts++
			for _, p := range r.Positions {
				rep.LossBans += p.Bans
				c.Positions++
				if p.Logprob <= zeroLogprob {
					c.Zero++
					continue
				}
				sums[k] -= p.Logprob
			}
			continue
		}
		k := key{it.Split, it.Form, it.Lang}
		c := rep_[k]
		if c == nil {
			c = &RepetitionCell{Split: it.Split, Form: it.Form, Lang: it.Lang, Periods: []int{}}
			rep_[k] = c
		}
		for _, p := range r.Probs {
			if p == nil {
				rep.TokensNoProb++
			}
		}
		limit := r.StopType == "limit"
		l := FixedRule.Detect(r.GeneratedTokens, limit)
		c.N++
		if !limit {
			c.EOS++
		}
		if l.Looping {
			c.Loops++
			starts[k] = append(starts[k], l.Start)
			c.Periods = append(c.Periods, l.Period)
		} else if limit {
			c.Censored++
		}
		score := ScoreAnswer(it.Answer, it.Form, r.Content)
		mentioned := AnswerMentioned(it.Answer, it.Form, r.Content)
		pr := PerRecord{ID: r.ID, Looping: l.Looping, Tokens: len(r.GeneratedTokens), Score: score, Mentioned: mentioned}
		if l.Looping {
			pr.Start, pr.Period = &l.Start, &l.Period
		}
		rep.PerRecord = append(rep.PerRecord, pr)
		if score != scoreNone {
			a := ans[k]
			if a == nil {
				a = &AnswerCell{Split: it.Split, Form: it.Form, Lang: it.Lang}
				ans[k] = a
			}
			a.Scored++
			if mentioned {
				a.Mentioned++
			}
			switch score {
			case scoreCorrect:
				a.Correct++
			case scoreWrong:
				a.Wrong++
			case scoreUnanswered:
				a.Unanswered++
			}
		}
		if it.Control != "" {
			row := ControlRow{ID: r.ID, Split: it.Split, Documented: it.Control, Looping: l.Looping, Agrees: l.Looping == (it.Control == controlLoop)}
			if l.Looping {
				row.Start, row.Period = &l.Start, &l.Period
				if p := r.Probs[l.Start]; p != nil {
					pf := math.Exp(p.Logprob)
					row.PFirst = &pf
				}
				var ps []float64
				for _, p := range r.Probs[l.Start+l.Period:] {
					if p != nil {
						ps = append(ps, math.Exp(p.Logprob))
					}
				}
				if len(ps) > 0 {
					pl := medianFloats(ps)
					row.PLoop = &pl
				}
			}
			rep.Controls = append(rep.Controls, row)
		}
	}
	for k, c := range rep_ {
		c.Rate = float64(c.Loops) / float64(c.N)
		if s := starts[k]; len(s) > 0 {
			m, latest := medianInts(s), slices.Max(s)
			c.MedianStart, c.MaxStart = &m, &latest
		}
		slices.Sort(c.Periods)
		rep.Repetition = append(rep.Repetition, *c)
	}
	sort.Slice(rep.Repetition, func(i, j int) bool {
		a, b := rep.Repetition[i], rep.Repetition[j]
		return cellLess(a.Split, a.Form, a.Lang, b.Split, b.Form, b.Lang)
	})
	for _, a := range ans {
		a.Accuracy = float64(a.Correct) / float64(a.Scored)
		a.MentionRate = float64(a.Mentioned) / float64(a.Scored)
		rep.Answers = append(rep.Answers, *a)
	}
	sort.Slice(rep.Answers, func(i, j int) bool {
		a, b := rep.Answers[i], rep.Answers[j]
		return cellLess(a.Split, a.Form, a.Lang, b.Split, b.Form, b.Lang)
	})
	for k, c := range loss {
		if c.Zero == 0 && c.Positions > 0 {
			nll := sums[k] / float64(c.Positions)
			ppl := math.Exp(nll)
			c.MeanNLL, c.PPL = &nll, &ppl
		}
		rep.Loss = append(rep.Loss, *c)
	}
	sort.Slice(rep.Loss, func(i, j int) bool {
		a, b := rep.Loss[i], rep.Loss[j]
		return cellLess(a.Split, formText, a.Lang, b.Split, formText, b.Lang)
	})
	return rep, nil
}

func optFloat(v *float64, format string) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf(format, *v)
}

func optInt(v *int) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprint(*v)
}

// libsDigest condenses the pinned libraries for the header. report.json has them
// one by one.
func libsDigest(libs map[string]string) string {
	var b strings.Builder
	for _, name := range slices.Sorted(maps.Keys(libs)) {
		fmt.Fprintf(&b, "%s %s\n", name, libs[name])
	}
	return sha256String(b.String())
}

// writeReport prints the tables as Markdown. Forms are never pooled.
func writeReport(w io.Writer, r *Report) {
	p := r.Pins
	fmt.Fprintf(w, "# Loop baseline report\n\n%d of %d records.\n\n", r.Records, r.Items)
	fmt.Fprintf(w, "- weight: %s, sha256 %s, %d bytes\n", p.WeightRealpath, p.WeightSHA256, p.WeightSize)
	fmt.Fprintf(w, "- server: %s, sha256 %s, %d libraries next to it, digest %s, build %s\n", p.ServerRealpath, p.ServerSHA256, len(p.ServerLibs), libsDigest(p.ServerLibs), p.ServerBuildInfo)
	if p.Attached {
		fmt.Fprint(w, "- server flags and environment: unknown, attached with -url; the binary above is what -server named, unverified\n")
	} else {
		fmt.Fprintf(w, "- server flags %v, environment %v\n", p.ServerArgs, p.ServerEnv)
	}
	fmt.Fprintf(w, "- chat template sha256 %s; corpus sha256 %s; vocabulary %d\n", p.ChatTemplateSHA256, p.CorpusSHA256, p.NVocab)
	fmt.Fprintf(w, "- greedy, seed %d, n_predict %d (%d with thinking on), n_probs %d\n", p.Settings.Seed, p.Settings.NPredict, p.Settings.NPredictThink, p.Settings.NProbs)
	fmt.Fprintf(w, "- generated tokens without a listed probability (incomplete UTF-8): %d; bans while measuring the loss: %d\n", r.TokensNoProb, r.LossBans)
	fmt.Fprintf(w, "- rule: period at most %d, at least %d whole units and %d tokens up to the last generated token, stopped by the limit\n", p.Settings.Rule.MaxPeriod, p.Settings.Rule.MinRepeats, p.Settings.Rule.MinRegion)
	fmt.Fprintf(w, "- censored by the limit: a loop of period p counts only if it starts by n_predict - max(%d, %d*p), and n_predict is %d, or %d with thinking on. The rate is a lower bound, starts cover detected loops only, and forms have different windows\n\n", p.Settings.Rule.MinRegion, p.Settings.Rule.MinRepeats, p.Settings.NPredict, p.Settings.NPredictThink)

	fmt.Fprint(w, "## Repetition\n\n| Split | Form | Lang | Prompts | Loops | Rate | Median start | Latest start | Periods | Ended by EOS | Limit, no loop (censored) |\n|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|\n")
	for _, c := range r.Repetition {
		fmt.Fprintf(w, "| %s | %s | %s | %d | %d | %.3f | %s | %s | %v | %d | %d |\n", c.Split, c.Form, c.Lang, c.N, c.Loops, c.Rate, optFloat(c.MedianStart, "%.1f"), optInt(c.MaxStart), c.Periods, c.EOS, c.Censored)
	}
	fmt.Fprint(w, "\n## Answer accuracy\n\nAccuracy scores the first sentence. Mentioned counts an accepted term anywhere, even after a wrong first answer: an occurrence rate that grows with length.\n\n| Split | Form | Lang | Scored | Correct | Wrong | Unanswered | Accuracy | Mentioned | Mention rate |\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, c := range r.Answers {
		fmt.Fprintf(w, "| %s | %s | %s | %d | %d | %d | %d | %.3f | %d | %.3f |\n", c.Split, c.Form, c.Lang, c.Scored, c.Correct, c.Wrong, c.Unanswered, c.Accuracy, c.Mentioned, c.MentionRate)
	}
	fmt.Fprint(w, "\n## Held-out loss\n\n| Split | Lang | Texts | Positions | Zero probability | Mean NLL (nats) | Perplexity |\n|---|---|---:|---:|---:|---:|---:|\n")
	for _, c := range r.Loss {
		fmt.Fprintf(w, "| %s | %s | %d | %d | %d | %s | %s |\n", c.Split, c.Lang, c.Texts, c.Positions, c.Zero, optFloat(c.MeanNLL, "%.4f"), optFloat(c.PPL, "%.3f"))
	}
	fmt.Fprint(w, "\n## Documented controls\n\n| Prompt | Split | Documented | Loops here | Start | Period | p(first) | p(in loop) | Agrees |\n|---|---|---|---|---:|---:|---:|---:|---|\n")
	for _, c := range r.Controls {
		fmt.Fprintf(w, "| %s | %s | %s | %v | %s | %s | %s | %s | %v |\n", c.ID, c.Split, c.Documented, c.Looping, optInt(c.Start), optInt(c.Period), optFloat(c.PFirst, "%.3f"), optFloat(c.PLoop, "%.3f"), c.Agrees)
	}
	fmt.Fprint(w, "\n## Not done\n\n")
	for _, s := range r.NotDone {
		fmt.Fprintf(w, "- %s\n", s)
	}
}
