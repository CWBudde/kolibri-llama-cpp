package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"slices"
	"sort"
	"strings"
)

// Forms of a corpus item. They are never pooled in a report: raw and chat
// behave differently (docs/real-checkpoint.md).
const (
	formRaw       = "raw"        // the prompt goes to the model as it is
	formChat      = "chat"       // the prompt is a user message, thinking off
	formChatThink = "chat-think" // the prompt is a user message, thinking on
	formText      = "text"       // not generated: the text whose held-out loss is measured
)

// Controls: what docs/real-checkpoint.md documents for a prompt on this weight.
const (
	controlLoop    = "loop"
	controlNonLoop = "non-loop"
)

// Strata of the split. Each pair belongs to one.
const (
	stratumLoop    = "documented-loop"
	stratumNonLoop = "documented-non-loop"
	stratumAnswer  = "answer-known"
	stratumOpen    = "open"
	stratumThink   = "thinking"
	stratumText    = "text"
)

// The two splits.
const (
	splitTuning  = "tuning"
	splitHeldOut = "held-out"
)

// splitSalt versions the split rule. Changing it, or anything else in the
// corpus, moves the pinned corpus hash in main_test.go.
const splitSalt = "kolibri-loops/split/v2/"

// Item is one line of the corpus.
type Item struct {
	ID   string `json:"id"`
	Pair string `json:"pair"`
	// Group holds every pair that shares a fact, a prompt or an answer with
	// another. The split assigns groups, so none of that crosses it.
	Group   string `json:"group"`
	Lang    string `json:"lang"`
	Form    string `json:"form"`
	Stratum string `json:"stratum"`
	Split   string `json:"split"`
	Prompt  string `json:"prompt"`
	// Answer is the known answer, if any.
	Answer *Answer `json:"answer,omitempty"`
	// Control and Source state what is documented for this prompt, and where.
	Control string `json:"control,omitempty"`
	Source  string `json:"source,omitempty"`
	// ExpectTokens, if set, must be what the server tokenizes the prompt into.
	ExpectTokens []int `json:"expect_tokens,omitempty"`
}

// Corpus is the parsed corpus and the hash of its bytes.
type Corpus struct {
	Items  []Item
	SHA256 string
}

var idPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9.-]*$`)

// loadCorpus reads and validates a corpus file.
func loadCorpus(path string) (*Corpus, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var items []Item
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 1<<16), 1<<24)
	for line := 1; sc.Scan(); line++ {
		if len(bytes.TrimSpace(sc.Bytes())) == 0 {
			continue
		}
		dec := json.NewDecoder(bytes.NewReader(sc.Bytes()))
		dec.DisallowUnknownFields()
		var it Item
		if err := dec.Decode(&it); err != nil {
			return nil, fmt.Errorf("%s:%d: %v", path, line, err)
		}
		items = append(items, it)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if err := validateCorpus(items); err != nil {
		return nil, fmt.Errorf("%s: %v", path, err)
	}
	h := sha256.Sum256(data)
	return &Corpus{Items: items, SHA256: hex.EncodeToString(h[:])}, nil
}

// cellsOf is what a pair of a stratum must contain, one item per (lang, form).
func cellsOf(stratum string) ([]string, error) {
	switch stratum {
	case stratumLoop, stratumNonLoop, stratumAnswer, stratumOpen:
		return []string{"de/" + formChat, "de/" + formRaw, "en/" + formChat, "en/" + formRaw}, nil
	case stratumThink:
		return []string{"de/" + formChatThink, "en/" + formChatThink}, nil
	case stratumText:
		return []string{"de/" + formText, "en/" + formText}, nil
	}
	return nil, fmt.Errorf("unknown stratum %q", stratum)
}

// validateCorpus checks the balance and that every split is what the rule says.
func validateCorpus(items []Item) error {
	if len(items) == 0 {
		return fmt.Errorf("empty corpus")
	}
	seen := map[string]bool{}
	pairCells := map[string][]string{}
	pairStratum := map[string]string{}
	pairGroup := map[string]string{}
	for _, it := range items {
		if !idPattern.MatchString(it.ID) {
			return fmt.Errorf("bad id %q", it.ID)
		}
		if seen[it.ID] {
			return fmt.Errorf("duplicate id %q", it.ID)
		}
		seen[it.ID] = true
		if it.Prompt == "" {
			return fmt.Errorf("%s: empty prompt", it.ID)
		}
		if it.Lang != "de" && it.Lang != "en" {
			return fmt.Errorf("%s: unknown lang %q", it.ID, it.Lang)
		}
		if want := it.Pair + "." + it.Lang + "." + it.Form; it.ID != want {
			return fmt.Errorf("%s: id should be %s", it.ID, want)
		}
		if s, ok := pairStratum[it.Pair]; ok && s != it.Stratum {
			return fmt.Errorf("%s: pair %s has strata %s and %s", it.ID, it.Pair, s, it.Stratum)
		}
		pairStratum[it.Pair] = it.Stratum
		if it.Group == "" {
			return fmt.Errorf("%s: no group", it.ID)
		}
		if g, ok := pairGroup[it.Pair]; ok && g != it.Group {
			return fmt.Errorf("%s: pair %s has groups %s and %s", it.ID, it.Pair, g, it.Group)
		}
		pairGroup[it.Pair] = it.Group
		pairCells[it.Pair] = append(pairCells[it.Pair], it.Lang+"/"+it.Form)
		if it.Control != "" && it.Control != controlLoop && it.Control != controlNonLoop {
			return fmt.Errorf("%s: unknown control %q", it.ID, it.Control)
		}
		if (it.Control != "") != (it.Source != "") {
			return fmt.Errorf("%s: a control needs a source, and a source a control", it.ID)
		}
		if it.Form == formText && it.Answer != nil {
			return fmt.Errorf("%s: a text has no answer", it.ID)
		}
		if it.Answer != nil && len(it.Answer.Accept) == 0 {
			return fmt.Errorf("%s: an answer without accepted terms", it.ID)
		}
	}
	for pair, cells := range pairCells {
		want, err := cellsOf(pairStratum[pair])
		if err != nil {
			return fmt.Errorf("pair %s: %v", pair, err)
		}
		slices.Sort(cells)
		if !slices.Equal(cells, want) {
			return fmt.Errorf("pair %s: has %v, a %s pair needs %v", pair, cells, pairStratum[pair], want)
		}
	}
	if err := checkGroups(items); err != nil {
		return err
	}
	splits := assignSplits(items)
	for _, it := range items {
		if it.Split != splits[it.Group] {
			return fmt.Errorf("%s: split is %q, the rule says %q", it.ID, it.Split, splits[it.Group])
		}
	}
	return nil
}

// checkGroups refuses what would leak across a split: items of different groups
// that share a prompt, or where one prompt contains the other, that accept the
// same term, or where one prompt holds the other's accepted term. It cannot see
// a fact restated in other words; the groups in the corpus handle those.
func checkGroups(items []Item) error {
	bare := func(it Item) string {
		return strings.ToLower(strings.ReplaceAll(it.Prompt, "<|endoftext|>", ""))
	}
	for _, a := range items {
		for _, b := range items {
			if a.Group == b.Group {
				continue
			}
			if strings.Contains(bare(a), bare(b)) {
				return fmt.Errorf("%s holds the prompt of %s, so they need one group", a.ID, b.ID)
			}
			if a.Answer == nil {
				continue
			}
			if firstTerm(b.Prompt, a.Answer.Accept) >= 0 {
				return fmt.Errorf("the prompt of %s holds the answer of %s, so they need one group", b.ID, a.ID)
			}
			if b.Answer != nil && slices.ContainsFunc(a.Answer.Accept, func(t string) bool {
				return slices.ContainsFunc(b.Answer.Accept, func(u string) bool { return strings.EqualFold(t, u) })
			}) {
				return fmt.Errorf("%s and %s accept the same answer, so they need one group", a.ID, b.ID)
			}
		}
	}
	return nil
}

// strataOrder ranks the strata: a group belongs to the first one its pairs have.
var strataOrder = []string{stratumLoop, stratumNonLoop, stratumAnswer, stratumOpen, stratumThink, stratumText}

// assignSplits is the split rule, from group to split. A whole group goes to one
// split, so a fact, a prompt or an answer never sits on both sides. Within a
// stratum the groups are ordered by the SHA-256 of salt and name and take turns,
// tuning first. With two groups holding documented loops, each split gets one.
func assignSplits(items []Item) map[string]string {
	stratum := map[string]string{}
	for _, it := range items {
		s, ok := stratum[it.Group]
		if !ok || rank(strataOrder, it.Stratum) < rank(strataOrder, s) {
			stratum[it.Group] = it.Stratum
		}
	}
	byStratum := map[string][]string{}
	for g, s := range stratum {
		byStratum[s] = append(byStratum[s], g)
	}
	out := map[string]string{}
	for _, groups := range byStratum {
		key := func(g string) string {
			h := sha256.Sum256([]byte(splitSalt + g))
			return hex.EncodeToString(h[:])
		}
		sort.Slice(groups, func(i, j int) bool { return key(groups[i]) < key(groups[j]) })
		for i, g := range groups {
			if i%2 == 0 {
				out[g] = splitTuning
			} else {
				out[g] = splitHeldOut
			}
		}
	}
	return out
}
