// Command kolibri-peek fetches individual (small) tensors from a Hugging
// Face safetensors checkpoint through HTTP range requests and prints value
// statistics. It is meant for sanity checks such as "are norm weights stored
// as w or as 1+w" without downloading shards.
//
//	kolibri-peek -repo Aleph-Alpha/Kolibri-1-BF16 -revision <sha> \
//	    model.layers.0.input_layernorm.weight model.layers.0.moe.router.expert_bias
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"slices"
	"strings"

	"kolibri-llm/internal/hf"
	"kolibri-llm/internal/safetensors"
)

// maxBytes refuses accidental multi-GB reads.
const maxBytes = 64 << 20

func main() {
	repo := flag.String("repo", "", "Hugging Face repo")
	revision := flag.String("revision", "", "full commit SHA")
	cmpRepo := flag.String("compare-repo", "", "optional second repo to diff against (e.g. the FP8 checkpoint)")
	cmpRevision := flag.String("compare-revision", "", "full commit SHA of -compare-repo")
	head := flag.Int("n", 8, "number of leading values to print")
	flag.Parse()
	if *repo == "" || len(*revision) != 40 || flag.NArg() == 0 || (*cmpRepo != "" && len(*cmpRevision) != 40) {
		flag.Usage()
		os.Exit(2)
	}
	ctx := context.Background()
	c := hf.NewClient()
	ck, err := openCheckpoint(ctx, c, *repo, *revision)
	if err != nil {
		log.Fatal(err)
	}
	var other *checkpoint
	if *cmpRepo != "" {
		if other, err = openCheckpoint(ctx, c, *cmpRepo, *cmpRevision); err != nil {
			log.Fatal(err)
		}
	}

	for _, name := range flag.Args() {
		t, vals, err := ck.load(ctx, name)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s %s%v\n  %s\n  head: %s\n", name, t.DType, t.Shape, stats(vals), fmtVals(vals[:min(*head, len(vals))]))
		if other == nil {
			continue
		}
		ot, ovals, err := other.load(ctx, name)
		if err != nil {
			log.Fatal(err)
		}
		if !slices.Equal(t.Shape, ot.Shape) {
			log.Fatalf("%s: shape %v vs %v", name, t.Shape, ot.Shape)
		}
		fmt.Printf("  vs %s (%s): %s\n", other.repo, ot.DType, diff(vals, ovals))
	}
}

type checkpoint struct {
	c         *hf.Client
	repo, rev string
	weightMap map[string]string
	headers   map[string]*safetensors.Header
}

func openCheckpoint(ctx context.Context, c *hf.Client, repo, rev string) (*checkpoint, error) {
	raw, err := c.Fetch(ctx, repo, rev, "model.safetensors.index.json")
	if err != nil {
		return nil, err
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(raw, &index); err != nil {
		return nil, err
	}
	return &checkpoint{c: c, repo: repo, rev: rev, weightMap: index.WeightMap, headers: map[string]*safetensors.Header{}}, nil
}

// raw fetches one tensor's payload and decodes it without dequantization.
func (ck *checkpoint) raw(ctx context.Context, name string) (safetensors.Tensor, []float64, error) {
	var t safetensors.Tensor
	shard, ok := ck.weightMap[name]
	if !ok {
		return t, nil, fmt.Errorf("%s: not in %s index", name, ck.repo)
	}
	r := ck.c.Open(ctx, ck.repo, ck.rev, shard)
	h := ck.headers[shard]
	if h == nil {
		var err error
		if h, err = safetensors.ReadHeader(r); err != nil {
			return t, nil, err
		}
		ck.headers[shard] = h
	}
	i := slices.IndexFunc(h.Tensors, func(t safetensors.Tensor) bool { return t.Name == name })
	if i < 0 {
		return t, nil, fmt.Errorf("%s: not in %s header", name, shard)
	}
	t = h.Tensors[i]
	if t.NumBytes() > maxBytes {
		return t, nil, fmt.Errorf("%s: %d bytes exceeds the %d byte limit", name, t.NumBytes(), maxBytes)
	}
	buf := make([]byte, t.NumBytes())
	if _, err := r.ReadAt(buf, 8+h.Size+t.Offsets[0]); err != nil {
		return t, nil, err
	}
	vals, err := decode(t.DType, buf)
	if err != nil {
		return t, nil, fmt.Errorf("%s: %w", name, err)
	}
	return t, vals, nil
}

// load returns a tensor's values, applying the FP8 block scale
// (<name>_scale_inv) when the checkpoint has one.
func (ck *checkpoint) load(ctx context.Context, name string) (safetensors.Tensor, []float64, error) {
	t, vals, err := ck.raw(ctx, name)
	if err != nil {
		return t, nil, err
	}
	scaleName := name + "_scale_inv"
	if _, ok := ck.weightMap[scaleName]; !ok {
		return t, vals, nil
	}
	st, scale, err := ck.raw(ctx, scaleName)
	if err != nil {
		return t, nil, err
	}
	if err := dequantBlocks(vals, t.Shape, scale, st.Shape); err != nil {
		return t, nil, fmt.Errorf("%s: %w", name, err)
	}
	return t, vals, nil
}

// dequantBlocks multiplies a [rows, cols] matrix in place by per-block
// scales of shape [ceil(rows/b0), ceil(cols/b1)].
func dequantBlocks(w []float64, shape []int64, scale []float64, sshape []int64) error {
	if len(shape) != 2 || len(sshape) != 2 {
		return fmt.Errorf("block scales need 2-D tensors, got %v and %v", shape, sshape)
	}
	rows, cols := shape[0], shape[1]
	b0, b1 := (rows+sshape[0]-1)/sshape[0], (cols+sshape[1]-1)/sshape[1]
	for i := range rows {
		for j := range cols {
			w[i*cols+j] *= scale[(i/b0)*sshape[1]+j/b1]
		}
	}
	return nil
}

func diff(a, b []float64) string {
	var maxAbs, maxRef, sq, ref float64
	for i := range a {
		d := math.Abs(a[i] - b[i])
		maxAbs = math.Max(maxAbs, d)
		maxRef = math.Max(maxRef, math.Abs(a[i]))
		sq += d * d
		ref += a[i] * a[i]
	}
	return fmt.Sprintf("max|d|=%.4g (max|ref|=%.4g) rel_rmse=%.4g", maxAbs, maxRef, math.Sqrt(sq/math.Max(ref, 1e-300)))
}

func decode(dtype string, b []byte) ([]float64, error) {
	var out []float64
	switch dtype {
	case "BF16":
		for i := 0; i+1 < len(b); i += 2 {
			out = append(out, float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[i:]))<<16)))
		}
	case "F32":
		for i := 0; i+3 < len(b); i += 4 {
			out = append(out, float64(math.Float32frombits(binary.LittleEndian.Uint32(b[i:]))))
		}
	case "F8_E4M3":
		for _, v := range b {
			out = append(out, e4m3(v))
		}
	default:
		return nil, fmt.Errorf("unsupported dtype %s", dtype)
	}
	return out, nil
}

// e4m3 decodes an OCP FP8 E4M3FN value (bias 7, no infinities, 0x7f/0xff NaN).
func e4m3(v byte) float64 {
	sign := 1.0
	if v&0x80 != 0 {
		sign = -1
	}
	exp, man := int(v>>3)&0xf, float64(v&7)
	switch {
	case exp == 0xf && man == 7:
		return math.NaN()
	case exp == 0:
		return sign * man / 8 * math.Ldexp(1, -6)
	default:
		return sign * (1 + man/8) * math.Ldexp(1, exp-7)
	}
}

func stats(v []float64) string {
	if len(v) == 0 {
		return "empty"
	}
	lo, hi, sum, sq := math.Inf(1), math.Inf(-1), 0.0, 0.0
	for _, x := range v {
		lo, hi = math.Min(lo, x), math.Max(hi, x)
		sum += x
		sq += x * x
	}
	mean := sum / float64(len(v))
	return fmt.Sprintf("n=%d min=%.6g max=%.6g mean=%.6g std=%.6g", len(v), lo, hi, mean, math.Sqrt(math.Max(0, sq/float64(len(v))-mean*mean)))
}

func fmtVals(v []float64) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = fmt.Sprintf("%.5g", x)
	}
	return strings.Join(parts, " ")
}
