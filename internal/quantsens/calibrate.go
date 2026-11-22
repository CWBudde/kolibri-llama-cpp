package quantsens

import (
	"fmt"
	"regexp"
)

// A single item's ΔKL against the BF16 logits is too noisy to rank items on
// 20 × 512 tokens: a change that alters a few router near-ties moves the
// mean KLD by a few thousandths either way, about as much as one band of
// layers adds. The KL of a candidate against the base file's own logits
// (the proxy) has no such floor: it is zero for the base and grows with the
// change. Calibrate turns proxy records into estimates of the ΔKL against
// BF16 by scaling them, per kind and type, with the ratio of the two
// measured for the whole kind. That assumes an item's share of its kind's
// ΔKL equals its share of the kind's proxy KL; an upgrade gets a negative
// ratio and so a negative estimate.

// Ratio is the scale Calibrate applies to one kind and type.
type Ratio struct {
	Kind, Type string
	Target     float64 // the kind's ΔKL against BF16
	Proxy      float64 // the kind's KL against the base
}

var layerItemRe = regexp.MustCompile(`^blk\.\d+\.(.+)$`)

// RecordKind is the kind of a record's item: blk.12.attn, blk.10-14.attn and
// attn are all attn; token_embd and output are their own kinds.
func RecordKind(item string) string {
	if _, _, k, ok := ParseBand(item); ok {
		return k
	}
	if m := layerItemRe.FindStringSubmatch(item); m != nil {
		return m[1]
	}
	return item
}

// Calibrate scales every non-base proxy record by the target/proxy ratio of
// its kind and type, from the kind-scope records of both. Records without
// both kind measurements, or whose kind's proxy KL is not positive, are
// dropped and counted.
func Calibrate(proxy, target []Record) (out []Record, ratios []Ratio, dropped int) {
	kindRec := func(recs []Record) map[[2]string]Record {
		m := map[[2]string]Record{}
		for _, r := range recs {
			if r.Scope == "kind" {
				m[[2]string{r.Item, r.Type}] = r
			}
		}
		return m
	}
	pk, tk := kindRec(proxy), kindRec(target)
	seen := map[[2]string]bool{}
	for _, r := range proxy {
		if r.Scope == "base" {
			continue
		}
		key := [2]string{RecordKind(r.Item), r.Type}
		p, okP := pk[key]
		t, okT := tk[key]
		if !okP || !okT || p.DKL <= 0 {
			dropped++
			continue
		}
		if !seen[key] {
			seen[key] = true
			ratios = append(ratios, Ratio{Kind: key[0], Type: key[1], Target: t.DKL, Proxy: p.DKL})
		}
		r.DKL *= t.DKL / p.DKL
		out = append(out, r)
	}
	return out, ratios, dropped
}

func (r Ratio) String() string {
	return fmt.Sprintf("%-14s %-7s target ΔKL %+.6f, proxy KL %.6f, scale %+.3f", r.Kind, r.Type, r.Target, r.Proxy, r.Target/r.Proxy)
}
