// Command kolibri-inventory builds a machine-readable tensor inventory of a
// Kolibri-1 checkpoint on Hugging Face without downloading tensor payloads.
//
// It fetches the small metadata files at a pinned revision, reads every
// shard's safetensors header through HTTP range requests, classifies each
// tensor, checks it against config.json, and writes:
//
//	<out>/config.json, generation_config.json, tokenizer_config.json  (verbatim)
//	<out>/summary.json        files, shards, per-class shapes/dtypes, checks
//	<out>/tensors.jsonl.gz    one line per tensor, with its GGUF target name
//
// It exits non-zero if any check fails.
package main

import (
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"sync"

	"kolibri-llm/internal/hf"
	"kolibri-llm/internal/kolibri"
	"kolibri-llm/internal/safetensors"
)

// verbatim are copied into the output directory; tokenizer.json (9 MB) is
// only hashed and summarized.
var verbatim = []string{"config.json", "generation_config.json", "tokenizer_config.json"}

const (
	indexFile     = "model.safetensors.index.json"
	tokenizerFile = "tokenizer.json"
)

func main() {
	repo := flag.String("repo", "", "Hugging Face repo, e.g. Aleph-Alpha/Kolibri-1-BF16")
	revision := flag.String("revision", "", "full commit SHA to pin (branch names are rejected)")
	out := flag.String("out", "", "output directory")
	jobs := flag.Int("j", 8, "concurrent header fetches")
	flag.Parse()
	if *repo == "" || *out == "" || len(*revision) != 40 {
		flag.Usage()
		os.Exit(2)
	}
	if err := run(context.Background(), *repo, *revision, *out, *jobs); err != nil {
		log.Fatal(err)
	}
}

// FileInfo identifies one fetched or referenced file.
type FileInfo struct {
	Name   string `json:"name"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256,omitempty"`
}

// ShardInfo summarizes one safetensors shard.
type ShardInfo struct {
	FileInfo
	HeaderBytes int64             `json:"header_bytes"`
	DataBytes   int64             `json:"data_bytes"`
	Tensors     int               `json:"tensors"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// Record is one line of tensors.jsonl.gz.
type Record struct {
	Name    string   `json:"name"`
	Shard   string   `json:"shard"`
	DType   string   `json:"dtype"`
	Shape   []int64  `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`
	Class   string   `json:"class"`
	Role    string   `json:"role"`
	Layer   int      `json:"layer"`
	Expert  int      `json:"expert"`
	GGUF    string   `json:"gguf,omitempty"`
}

// ClassSummary aggregates all tensors of one (class, role).
type ClassSummary struct {
	Class     string         `json:"class"`
	Group     kolibri.Group  `json:"group"`
	Role      kolibri.Role   `json:"role"`
	HF        string         `json:"hf"`
	GGUF      string         `json:"gguf,omitempty"`
	Count     int            `json:"count"`
	DTypes    map[string]int `json:"dtypes"`
	HFShape   []int64        `json:"hf_shape"`
	GGUFShape []int64        `json:"gguf_shape,omitempty"`
	Bytes     int64          `json:"bytes"`
	Note      string         `json:"note,omitempty"`
}

// Summary is summary.json.
type Summary struct {
	Repo       string          `json:"repo"`
	Revision   string          `json:"revision"`
	Files      []FileInfo      `json:"files"`
	Shards     []ShardInfo     `json:"shards"`
	Tensors    int             `json:"tensors"`
	Parameters int64           `json:"parameters"`
	DataBytes  int64           `json:"data_bytes"`
	IndexTotal int64           `json:"index_total_size"`
	Attention  AttentionLayout `json:"attention"`
	Classes    []*ClassSummary `json:"classes"`
	Tokenizer  *TokenizerInfo  `json:"tokenizer"`
	Errors     []string        `json:"errors"`
}

// AttentionLayout records the per-layer attention pattern from config.json.
type AttentionLayout struct {
	SWAPeriod     int   `json:"swa_period"`
	SlidingWindow int64 `json:"sliding_window"`
	FullLayers    []int `json:"full_attention_layers"`
}

func run(ctx context.Context, repo, revision, out string, jobs int) error {
	c := hf.NewClient()
	info, err := c.Info(ctx, repo, revision)
	if err != nil {
		return err
	}
	if info.SHA != revision {
		return fmt.Errorf("revision %s resolved to %s", revision, info.SHA)
	}
	siblings := map[string]hf.Sibling{}
	for _, s := range info.Siblings {
		siblings[s.Name] = s
	}

	sum := &Summary{Repo: repo, Revision: revision, Errors: []string{}}
	fail := func(format string, args ...any) { sum.Errors = append(sum.Errors, fmt.Sprintf(format, args...)) }

	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	files := map[string][]byte{}
	for _, name := range append(slices.Clone(verbatim), tokenizerFile, indexFile) {
		data, err := c.Fetch(ctx, repo, revision, name)
		if err != nil {
			return err
		}
		files[name] = data
		h := sha256.Sum256(data)
		sum.Files = append(sum.Files, FileInfo{Name: name, Size: int64(len(data)), SHA256: hex.EncodeToString(h[:])})
		if s, ok := siblings[name]; ok && s.Size != int64(len(data)) {
			fail("%s: downloaded %d bytes, repo lists %d", name, len(data), s.Size)
		}
	}
	for _, name := range verbatim {
		if err := os.WriteFile(filepath.Join(out, name), files[name], 0o644); err != nil {
			return err
		}
	}

	cfg, err := kolibri.ParseConfig(files["config.json"])
	if err != nil {
		return err
	}
	sum.Attention = AttentionLayout{SWAPeriod: cfg.SWAPeriod(), SlidingWindow: cfg.SlidingWindow}
	for i := range cfg.LayerTypes {
		if cfg.IsFullAttention(i) {
			sum.Attention.FullLayers = append(sum.Attention.FullLayers, i)
		}
	}
	if sum.Tokenizer, err = summarizeTokenizer(files[tokenizerFile], int(cfg.VocabSize)); err != nil {
		return err
	}

	var index struct {
		Metadata struct {
			TotalSize int64 `json:"total_size"`
		} `json:"metadata"`
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(files[indexFile], &index); err != nil {
		return fmt.Errorf("decode %s: %w", indexFile, err)
	}
	sum.IndexTotal = index.Metadata.TotalSize
	shardNames := slices.Sorted(maps.Values(index.WeightMap))
	shardNames = slices.Compact(shardNames)

	headers, err := fetchHeaders(ctx, c, repo, revision, shardNames, jobs)
	if err != nil {
		return err
	}

	var records []Record
	seen := map[string]bool{}
	classes := map[string]*ClassSummary{}
	for _, shard := range shardNames {
		h := headers[shard]
		si := ShardInfo{FileInfo: FileInfo{Name: shard, Size: h.size}, HeaderBytes: h.Size, DataBytes: h.DataSize(), Tensors: len(h.Tensors), Metadata: h.Metadata}
		if s, ok := siblings[shard]; !ok {
			fail("%s: not listed in repo", shard)
		} else {
			if s.LFS != nil {
				si.SHA256 = s.LFS.SHA256
			}
			if s.Size != h.FileSize() {
				fail("%s: header implies %d bytes, repo lists %d", shard, h.FileSize(), s.Size)
			}
		}
		if h.size != h.FileSize() {
			fail("%s: header implies %d bytes, server reports %d", shard, h.FileSize(), h.size)
		}
		sum.Shards = append(sum.Shards, si)

		for _, t := range h.Tensors {
			if seen[t.Name] {
				fail("%s: duplicate tensor in %s", t.Name, shard)
			}
			seen[t.Name] = true
			if want := index.WeightMap[t.Name]; want != shard {
				fail("%s: found in %s, index says %q", t.Name, shard, want)
			}
			sum.Tensors++
			sum.DataBytes += t.NumBytes()

			e, err := kolibri.Classify(t.Name)
			if err != nil {
				fail("%v", err)
				continue
			}
			if e.Layer >= cfg.NumHiddenLayers || int64(e.Expert) >= cfg.NumExperts {
				fail("%s: layer/expert index out of range", t.Name)
				continue
			}
			if want := cfg.ExpectedShape(e); !slices.Equal(t.Shape, want) {
				fail("%s: shape %v, want %v", t.Name, t.Shape, want)
			}
			if e.Role == kolibri.RoleWeight {
				sum.Parameters += t.NumElements()
			}
			records = append(records, Record{
				Name: t.Name, Shard: shard, DType: t.DType, Shape: t.Shape, Offsets: t.Offsets,
				Class: e.Spec.Class, Role: string(e.Role), Layer: e.Layer, Expert: e.Expert, GGUF: e.GGUFName(),
			})

			key := e.Spec.Class + "/" + string(e.Role)
			cs := classes[key]
			if cs == nil {
				cs = &ClassSummary{Class: e.Spec.Class, Group: e.Spec.Group, Role: e.Role, HF: e.Spec.HF, DTypes: map[string]int{}, HFShape: cfg.ExpectedShape(e), Note: e.Spec.Note}
				if e.Role == kolibri.RoleWeight {
					cs.GGUF, cs.GGUFShape = e.Spec.GGUF, e.Spec.GGUFShape(cfg)
				}
				classes[key] = cs
			}
			cs.Count++
			cs.DTypes[t.DType]++
			cs.Bytes += t.NumBytes()
		}
	}
	for name := range index.WeightMap {
		if !seen[name] {
			fail("%s: in index but in no shard header", name)
		}
	}
	if sum.IndexTotal != sum.DataBytes {
		fail("index total_size %d, headers sum to %d", sum.IndexTotal, sum.DataBytes)
	}
	var missing []string
	for _, name := range cfg.ExpectedNames() {
		if !seen[name] {
			missing = append(missing, name)
		}
		delete(seen, name)
	}
	for _, name := range missing {
		fail("%s: expected but missing", name)
	}
	for _, name := range slices.Sorted(maps.Keys(seen)) {
		fail("%s: present but not expected for this config", name)
	}
	for _, cs := range classes {
		if len(cs.DTypes) > 1 {
			fail("%s/%s: mixed dtypes %v", cs.Class, cs.Role, cs.DTypes)
		}
	}

	for _, s := range kolibri.Specs {
		for _, r := range []kolibri.Role{kolibri.RoleWeight, kolibri.RoleScaleInv} {
			if cs := classes[s.Class+"/"+string(r)]; cs != nil {
				sum.Classes = append(sum.Classes, cs)
			}
		}
	}

	if err := writeRecords(filepath.Join(out, "tensors.jsonl.gz"), records); err != nil {
		return err
	}
	if err := writeJSON(filepath.Join(out, "summary.json"), sum); err != nil {
		return err
	}
	log.Printf("%s@%s: %d tensors, %d parameters, %d bytes in %d shards, %d errors",
		repo, revision[:8], sum.Tensors, sum.Parameters, sum.DataBytes, len(sum.Shards), len(sum.Errors))
	if len(sum.Errors) > 0 {
		for _, e := range sum.Errors[:min(len(sum.Errors), 20)] {
			log.Print(e)
		}
		return errors.New("inventory checks failed")
	}
	return nil
}

type shardHeader struct {
	*safetensors.Header
	size int64 // from Content-Range
}

func fetchHeaders(ctx context.Context, c *hf.Client, repo, revision string, shards []string, jobs int) (map[string]shardHeader, error) {
	var (
		mu   sync.Mutex
		wg   sync.WaitGroup
		errs []error
		out  = map[string]shardHeader{}
		sem  = make(chan struct{}, jobs)
	)
	for _, shard := range shards {
		wg.Go(func() {
			sem <- struct{}{}
			defer func() { <-sem }()
			r := c.Open(ctx, repo, revision, shard)
			h, err := safetensors.ReadHeader(r)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				errs = append(errs, fmt.Errorf("%s: %w", shard, err))
				return
			}
			out[shard] = shardHeader{h, r.Size}
		})
	}
	wg.Wait()
	return out, errors.Join(errs...)
}

func writeRecords(path string, records []Record) error {
	sort.Slice(records, func(i, j int) bool {
		a, b := records[i], records[j]
		if a.Shard != b.Shard {
			return a.Shard < b.Shard
		}
		return a.Offsets[0] < b.Offsets[0]
	})
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	// Fixed header fields keep the gzip output reproducible.
	zw := gzip.NewWriter(f)
	enc := json.NewEncoder(zw)
	for _, r := range records {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	if err := zw.Close(); err != nil {
		return err
	}
	return f.Close()
}

func writeJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

// TokenizerInfo summarizes tokenizer.json for the Phase 2 compatibility work.
type TokenizerInfo struct {
	SHA256        string          `json:"sha256"`
	ModelType     string          `json:"model_type"`
	ByteFallback  bool            `json:"byte_fallback"`
	Vocab         int             `json:"vocab"`
	Merges        int             `json:"merges"`
	MergesFormat  string          `json:"merges_format"`
	Normalizer    json.RawMessage `json:"normalizer"`
	PreTokenizer  json.RawMessage `json:"pre_tokenizer"`
	PostProcessor json.RawMessage `json:"post_processor"`
	Decoder       json.RawMessage `json:"decoder"`
	AddedTokens   []AddedToken    `json:"added_tokens"`
	// FileIDGaps are IDs below the largest ID in tokenizer.json that the file
	// leaves unused. They are not the runtime gaps; see AddedToken.RuntimeID.
	FileIDGaps []int `json:"file_id_gaps"`
	// RuntimeUnassigned are the IDs below config.json's vocab_size that no
	// token has once the tokenizers library has loaded the file.
	RuntimeUnassigned []int `json:"runtime_unassigned"`
}

// AddedToken is one entry of tokenizer.json's added_tokens.
type AddedToken struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
	// RuntimeID is the ID the HF tokenizers library actually assigns. It
	// ignores the "id" field: a token missing from the model vocab gets
	// max(len(vocab), largest added ID so far + 1), in file order.
	RuntimeID int `json:"runtime_id"`
}

func summarizeTokenizer(raw []byte, vocabSize int) (*TokenizerInfo, error) {
	var t struct {
		Model struct {
			Type         string            `json:"type"`
			ByteFallback bool              `json:"byte_fallback"`
			Vocab        map[string]int    `json:"vocab"`
			Merges       []json.RawMessage `json:"merges"`
		} `json:"model"`
		Normalizer    json.RawMessage `json:"normalizer"`
		PreTokenizer  json.RawMessage `json:"pre_tokenizer"`
		PostProcessor json.RawMessage `json:"post_processor"`
		Decoder       json.RawMessage `json:"decoder"`
		AddedTokens   []AddedToken    `json:"added_tokens"`
	}
	if err := json.Unmarshal(raw, &t); err != nil {
		return nil, fmt.Errorf("decode %s: %w", tokenizerFile, err)
	}
	h := sha256.Sum256(raw)
	info := &TokenizerInfo{
		SHA256: hex.EncodeToString(h[:]), ModelType: t.Model.Type, ByteFallback: t.Model.ByteFallback,
		Vocab: len(t.Model.Vocab), Merges: len(t.Model.Merges),
		Normalizer: t.Normalizer, PreTokenizer: t.PreTokenizer, PostProcessor: t.PostProcessor, Decoder: t.Decoder,
		AddedTokens: t.AddedTokens, FileIDGaps: []int{}, RuntimeUnassigned: []int{},
	}
	if len(t.Model.Merges) > 0 {
		info.MergesFormat = "pairs"
		if strings.HasPrefix(string(t.Model.Merges[0]), `"`) {
			info.MergesFormat = "strings"
		}
	}
	info.FileIDGaps = idGaps(t.Model.Vocab, t.AddedTokens, func(a AddedToken) int { return a.ID }, -1)
	assignRuntimeIDs(t.Model.Vocab, info.AddedTokens)
	info.RuntimeUnassigned = idGaps(t.Model.Vocab, info.AddedTokens, func(a AddedToken) int { return a.RuntimeID }, vocabSize-1)
	return info, nil
}

// assignRuntimeIDs mirrors AddedVocabulary::add_tokens in HF tokenizers.
func assignRuntimeIDs(vocab map[string]int, added []AddedToken) {
	next := len(vocab)
	for i := range added {
		if id, ok := vocab[added[i].Content]; ok {
			added[i].RuntimeID = id
			continue
		}
		added[i].RuntimeID = next
		next++
	}
}

// idGaps lists the IDs in [0, upTo] that no token uses; upTo < 0 means up to
// the largest used ID.
func idGaps(vocab map[string]int, added []AddedToken, id func(AddedToken) int, upTo int) []int {
	used := map[int]bool{}
	for _, v := range vocab {
		used[v] = true
	}
	for _, a := range added {
		used[id(a)] = true
	}
	if upTo < 0 && len(used) > 0 {
		upTo = slices.Max(slices.Collect(maps.Keys(used)))
	}
	gaps := []int{}
	for i := 0; i <= upTo; i++ {
		if !used[i] {
			gaps = append(gaps, i)
		}
	}
	return gaps
}
