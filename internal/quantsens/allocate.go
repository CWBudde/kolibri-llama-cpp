package quantsens

import (
	"fmt"
	"maps"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// Score is what llama-perplexity --kl-divergence prints at the end.
type Score struct {
	KLD, KLDErr         float64
	SameTop, SameTopErr float64 // percent
	LnPPLRatio          float64
}

var (
	kldRe  = regexp.MustCompile(`Mean\s+KLD:\s*(-?[\d.]+)\s*±\s*([\d.]+)`)
	topRe  = regexp.MustCompile(`Same top p:\s*([\d.]+)\s*±\s*([\d.]+)\s*%`)
	lnppRe = regexp.MustCompile(`Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*(-?[\d.]+)`)
)

// ParseKLD reads the summary of a llama-perplexity --kl-divergence run.
func ParseKLD(out string) (Score, error) {
	var s Score
	m := kldRe.FindStringSubmatch(out)
	if m == nil {
		return s, fmt.Errorf("no \"Mean KLD\" line in the llama-perplexity output")
	}
	s.KLD, _ = strconv.ParseFloat(m[1], 64)
	s.KLDErr, _ = strconv.ParseFloat(m[2], 64)
	if m := topRe.FindStringSubmatch(out); m != nil {
		s.SameTop, _ = strconv.ParseFloat(m[1], 64)
		s.SameTopErr, _ = strconv.ParseFloat(m[2], 64)
	}
	if m := lnppRe.FindStringSubmatch(out); m != nil {
		s.LnPPLRatio, _ = strconv.ParseFloat(m[1], 64)
	}
	return s, nil
}

// Record is one measurement, a line of the measure step's JSONL. Scope
// "base" is the base file itself (Item and Type empty); "layer" and "kind"
// name the item's scope.
type Record struct {
	Scope      string  `json:"scope"`
	Item       string  `json:"item,omitempty"`
	Type       string  `json:"type,omitempty"`
	Bytes      int64   `json:"bytes,omitempty"` // the item's bytes in Type
	BaseType   string  `json:"base_type,omitempty"`
	BaseBytes  int64   `json:"base_bytes,omitempty"`
	KLD        float64 `json:"kld"`
	KLDErr     float64 `json:"kld_err"`
	SameTop    float64 `json:"same_top"`
	LnPPLRatio float64 `json:"ln_ppl_ratio"`
	DKL        float64 `json:"dkl"` // KLD minus the base file's KLD
	Seconds    float64 `json:"seconds"`
	Log        string  `json:"log,omitempty"`
}

// Option is one type for an item: its bytes and the KL it adds over the
// base. Derived marks a cost spread from a kind-scope measurement.
type Option struct {
	Type    string  `json:"type"`
	Bytes   int64   `json:"bytes"`
	DKL     float64 `json:"dkl"`
	Derived bool    `json:"derived,omitempty"`
}

// Choice is an item and its options; Options[0] is the base's type, at no
// added KL.
type Choice struct {
	Item    string
	Options []Option
}

// ItemInfo is what BuildChoices needs about an item: its kind, and its type
// and bytes in the base and in every donor that holds it whole in one type.
type ItemInfo struct {
	Item      Item
	BaseType  string
	BaseBytes int64
	Donor     map[string]int64 // type -> bytes
}

// BuildChoices makes each item's options from the measurements: a
// layer-scope record for the item and type if there is one, else a band
// record (scope band:N) covering its layer, else the kind-scope record for
// its kind and type, the band's or kind's added KL spread evenly over its
// items (assuming the per-layer costs sum to it), else no option for that
// type.
func BuildChoices(infos []ItemInfo, recs []Record) []Choice {
	layer := map[[2]string]Record{}
	kind := map[[2]string]Record{}
	type bandKey struct {
		kind, typ string
		layer     int
	}
	band := map[bandKey]float64{} // per-layer share of a band's added KL
	for _, r := range recs {
		switch {
		case r.Scope == "layer":
			layer[[2]string{r.Item, r.Type}] = r
		case r.Scope == "kind":
			kind[[2]string{r.Item, r.Type}] = r
		case strings.HasPrefix(r.Scope, "band:"):
			first, last, k, ok := ParseBand(r.Item)
			if !ok { // token_embd, output: whole items in any scope but kind
				layer[[2]string{r.Item, r.Type}] = r
				continue
			}
			for l := first; l <= last; l++ {
				band[bandKey{k, r.Type, l}] = r.DKL / float64(last-first+1)
			}
		}
	}
	perKind := map[string]int{}
	for _, in := range infos {
		perKind[in.Item.Kind]++
	}
	var out []Choice
	for _, in := range infos {
		c := Choice{Item: in.Item.Name, Options: []Option{{Type: in.BaseType, Bytes: in.BaseBytes}}}
		for _, typ := range slices.Sorted(maps.Keys(in.Donor)) {
			if typ == in.BaseType {
				continue
			}
			if r, ok := layer[[2]string{in.Item.Name, typ}]; ok {
				c.Options = append(c.Options, Option{Type: typ, Bytes: in.Donor[typ], DKL: r.DKL})
			} else if d, ok := band[bandKey{in.Item.Kind, typ, in.Item.Layer}]; ok && in.Item.Layer >= 0 {
				c.Options = append(c.Options, Option{Type: typ, Bytes: in.Donor[typ], DKL: d, Derived: true})
			} else if r, ok := kind[[2]string{in.Item.Kind, typ}]; ok {
				c.Options = append(c.Options, Option{Type: typ, Bytes: in.Donor[typ], DKL: r.DKL / float64(perKind[in.Item.Kind]), Derived: true})
			}
		}
		out = append(out, c)
	}
	return out
}

// Pick is the type a plan gives an item.
type Pick struct {
	Item string `json:"item"`
	Option
}

// Plan is an allocation: Fixed is the bytes of the tensors no item covers
// (router, norms), Total the whole file's tensor bytes, DKL the predicted
// added KL over the base.
type Plan struct {
	Budget int64   `json:"budget"`
	Fixed  int64   `json:"fixed"`
	Total  int64   `json:"total"`
	DKL    float64 `json:"dkl"`
	Picks  []Pick  `json:"picks"`
}

// Allocate picks one option per item with the least summed added KL whose
// bytes plus fixed fit in budget: a multiple-choice knapsack, solved exactly
// by dynamic programming over the bytes in units of unit (each option's
// size rounded up, so the plan never exceeds the budget).
func Allocate(choices []Choice, fixed, budget, unit int64) (Plan, error) {
	if unit <= 0 {
		return Plan{}, fmt.Errorf("unit %d", unit)
	}
	capU := (budget - fixed) / unit
	if capU < 0 {
		return Plan{}, fmt.Errorf("budget %d is below the fixed tensors' %d bytes", budget, fixed)
	}
	w := func(b int64) int64 { return (b + unit - 1) / unit }
	inf := math.Inf(1)
	prev := make([]float64, capU+1) // least cost at weight <= c with no items yet: 0
	cur := make([]float64, capU+1)
	pick := make([][]uint8, len(choices))
	for i, ch := range choices {
		if len(ch.Options) == 0 || len(ch.Options) > 255 {
			return Plan{}, fmt.Errorf("%s: %d options", ch.Item, len(ch.Options))
		}
		pick[i] = make([]uint8, capU+1)
		for c := int64(0); c <= capU; c++ {
			best, bo := inf, 0
			for o, op := range ch.Options {
				wo := w(op.Bytes)
				if wo > c {
					continue
				}
				if v := prev[c-wo] + op.DKL; v < best {
					best, bo = v, o
				}
			}
			cur[c], pick[i][c] = best, uint8(bo)
		}
		prev, cur = cur, prev
	}
	if math.IsInf(prev[capU], 1) {
		return Plan{}, fmt.Errorf("no plan fits %d bytes (fixed %d)", budget, fixed)
	}
	p := Plan{Budget: budget, Fixed: fixed, Total: fixed, Picks: make([]Pick, len(choices))}
	c := capU
	for i := len(choices) - 1; i >= 0; i-- {
		op := choices[i].Options[pick[i][c]]
		p.Picks[i] = Pick{Item: choices[i].Item, Option: op}
		p.Total += op.Bytes
		p.DKL += op.DKL
		c -= w(op.Bytes)
	}
	return p, nil
}

// TensorTypes writes a llama-quantize --tensor-type-file for a plan: one
// anchored pattern per tensor of every item, so that quantizing the BF16
// file with it reproduces the plan whatever the default type.
func TensorTypes(p Plan, items []Item) (string, error) {
	byName := map[string]Item{}
	for _, it := range items {
		byName[it.Name] = it
	}
	var b strings.Builder
	for _, pk := range p.Picks {
		it, ok := byName[pk.Item]
		if !ok {
			return "", fmt.Errorf("plan item %s is not an item of the file", pk.Item)
		}
		for _, name := range it.Tensors {
			fmt.Fprintf(&b, "^%s$=%s\n", regexp.QuoteMeta(name), strings.ToLower(pk.Type))
		}
	}
	return b.String(), nil
}
