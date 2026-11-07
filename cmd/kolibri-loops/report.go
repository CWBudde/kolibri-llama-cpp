package main

import (
	"encoding/json"
	"fmt"
	"io"
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
}

// RepetitionCell is one (split, form, lang) of the repetition table.
type RepetitionCell struct {
	Split       string  `json:"split"`
	Form        string  `json:"form"`
	Lang        string  `json:"lang"`
	N           int     `json:"n"`
	Loops       int     `json:"loops"`
	Rate        float64 `json:"rate"`
	MedianStart float64 `json:"median_start"`
	MaxStart    int     `json:"max_start"`
	Periods     []int   `json:"periods"`
	EOS         int     `json:"eos"`
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
	Start      int    `json:"start"`
	Period     int    `json:"period"`
	// PFirst is the probability of the first token of the loop, PLoop the median
	// probability of the generated tokens after the first unit.
	PFirst *float64 `json:"p_first"`
	PLoop  *float64 `json:"p_loop"`
	Agrees bool     `json:"agrees"`
}

// PerRecord is the detector's result for one record.
type PerRecord struct {
	ID     string `json:"id"`
	Loop   Loop   `json:"loop"`
	Tokens int    `json:"tokens"`
	Score  string `json:"score,omitempty"`
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
	if len(v) == 0 {
		return 0
	}
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

// checkRecord re-applies the checks of a run, so a hand-edited or truncated
// record does not reach the tables.
func checkRecord(r Record) error {
	if r.Item.ID != r.ID {
		return fmt.Errorf("record %s holds item %s", r.ID, r.Item.ID)
	}
	if r.Item.Form == formText {
		if len(r.Positions) != len(r.PromptTokens)-1 {
			return fmt.Errorf("record %s: %d positions for %d tokens", r.ID, len(r.Positions), len(r.PromptTokens))
		}
		for i, p := range r.Positions {
			if p.Target != r.PromptTokens[i+1] {
				return fmt.Errorf("record %s: position %d targets %d, the text has %d", r.ID, i+1, p.Target, r.PromptTokens[i+1])
			}
		}
		return nil
	}
	var req completionRequest
	if err := json.Unmarshal(r.Request, &req); err != nil {
		return fmt.Errorf("record %s: request: %v", r.ID, err)
	}
	if !slices.Equal(req.Prompt, r.PromptTokens) {
		return fmt.Errorf("record %s: the request's prompt is not prompt_tokens", r.ID)
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
	resp := completionResponse{Tokens: r.GeneratedTokens, StopType: r.StopType, TokensEvaluated: r.TokensEvaluated, TokensPredicted: len(r.GeneratedTokens), Probs: listed, GenerationSettings: r.ResolvedSettings}
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
			c.MaxStart = max(c.MaxStart, l.Start)
			c.Periods = append(c.Periods, l.Period)
		}
		score := ScoreAnswer(it.Answer, it.Form, r.Content)
		rep.PerRecord = append(rep.PerRecord, PerRecord{ID: r.ID, Loop: l, Tokens: len(r.GeneratedTokens), Score: score})
		if score != scoreNone {
			a := ans[k]
			if a == nil {
				a = &AnswerCell{Split: it.Split, Form: it.Form, Lang: it.Lang}
				ans[k] = a
			}
			a.Scored++
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
			row := ControlRow{ID: r.ID, Split: it.Split, Documented: it.Control, Looping: l.Looping, Start: l.Start, Period: l.Period, Agrees: l.Looping == (it.Control == controlLoop)}
			if l.Looping {
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
		c.MedianStart = medianInts(starts[k])
		slices.Sort(c.Periods)
		rep.Repetition = append(rep.Repetition, *c)
	}
	sort.Slice(rep.Repetition, func(i, j int) bool {
		a, b := rep.Repetition[i], rep.Repetition[j]
		return cellLess(a.Split, a.Form, a.Lang, b.Split, b.Form, b.Lang)
	})
	for _, a := range ans {
		a.Accuracy = float64(a.Correct) / float64(a.Scored)
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

// writeReport prints the tables as Markdown. Forms are never pooled.
func writeReport(w io.Writer, r *Report) {
	p := r.Pins
	fmt.Fprintf(w, "# Loop baseline report\n\n%d of %d records.\n\n", r.Records, r.Items)
	fmt.Fprintf(w, "- weight: %s, sha256 %s, %d bytes\n", p.WeightRealpath, p.WeightSHA256, p.WeightSize)
	fmt.Fprintf(w, "- server: %s, sha256 %s, build %s, args %v, attached %v\n", p.ServerRealpath, p.ServerSHA256, p.ServerBuildInfo, p.ServerArgs, p.Attached)
	fmt.Fprintf(w, "- chat template sha256 %s; corpus sha256 %s; vocabulary %d\n", p.ChatTemplateSHA256, p.CorpusSHA256, p.NVocab)
	fmt.Fprintf(w, "- greedy, seed %d, n_predict %d (%d with thinking on), n_probs %d\n", p.Settings.Seed, p.Settings.NPredict, p.Settings.NPredictThink, p.Settings.NProbs)
	fmt.Fprintf(w, "- generated tokens without a listed probability (incomplete UTF-8): %d; bans while measuring the loss: %d\n", r.TokensNoProb, r.LossBans)
	fmt.Fprintf(w, "- rule: period at most %d, at least %d whole units and %d tokens up to the last generated token, stopped by the limit\n\n", p.Settings.Rule.MaxPeriod, p.Settings.Rule.MinRepeats, p.Settings.Rule.MinRegion)

	fmt.Fprint(w, "## Repetition\n\n| Split | Form | Lang | Prompts | Loops | Rate | Median start | Latest start | Periods | Ended by EOS |\n|---|---|---|---:|---:|---:|---:|---:|---|---:|\n")
	for _, c := range r.Repetition {
		fmt.Fprintf(w, "| %s | %s | %s | %d | %d | %.3f | %.1f | %d | %v | %d |\n", c.Split, c.Form, c.Lang, c.N, c.Loops, c.Rate, c.MedianStart, c.MaxStart, c.Periods, c.EOS)
	}
	fmt.Fprint(w, "\n## Answer accuracy\n\n| Split | Form | Lang | Scored | Correct | Wrong | Unanswered | Accuracy |\n|---|---|---|---:|---:|---:|---:|---:|\n")
	for _, c := range r.Answers {
		fmt.Fprintf(w, "| %s | %s | %s | %d | %d | %d | %d | %.3f |\n", c.Split, c.Form, c.Lang, c.Scored, c.Correct, c.Wrong, c.Unanswered, c.Accuracy)
	}
	fmt.Fprint(w, "\n## Held-out loss\n\n| Split | Lang | Texts | Positions | Zero probability | Mean NLL (nats) | Perplexity |\n|---|---|---:|---:|---:|---:|---:|\n")
	for _, c := range r.Loss {
		fmt.Fprintf(w, "| %s | %s | %d | %d | %d | %s | %s |\n", c.Split, c.Lang, c.Texts, c.Positions, c.Zero, optFloat(c.MeanNLL, "%.4f"), optFloat(c.PPL, "%.3f"))
	}
	fmt.Fprint(w, "\n## Documented controls\n\n| Prompt | Split | Documented | Loops here | Start | Period | p(first) | p(in loop) | Agrees |\n|---|---|---|---|---:|---:|---:|---:|---|\n")
	for _, c := range r.Controls {
		fmt.Fprintf(w, "| %s | %s | %s | %v | %d | %d | %s | %s | %v |\n", c.ID, c.Split, c.Documented, c.Looping, c.Start, c.Period, optFloat(c.PFirst, "%.3f"), optFloat(c.PLoop, "%.3f"), c.Agrees)
	}
	fmt.Fprint(w, "\n## Not done\n\n")
	for _, s := range r.NotDone {
		fmt.Fprintf(w, "- %s\n", s)
	}
}
