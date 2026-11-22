// Command kolibri-qsens measures how much KL divergence each tensor group of
// a Kolibri GGUF adds in each llama.cpp type, and allocates the types under a
// byte budget, after LookAtMyQuant's llama.cpp route
// (github.com/ShinMegamiBoson/lookatmyquant, gguf_compiler.py, MIT). See
// internal/quantsens and PLAN.md Phase 7.
//
// Donors are GGUFs of the same model that each hold tensors in other types,
// typically one llama-quantize --pure file per type; the type of a group in
// a donor is read from its header, so donors need no labels.
//
//	kolibri-qsens splice   -base B.gguf -out X.gguf -take D.gguf=REGEX [-take ...]
//	kolibri-qsens measure  -base B.gguf -donors D1.gguf,D2.gguf -scope kind|layer \
//	                       -perplexity llama-perplexity -kld-base base.bin -work DIR -out sens.jsonl
//	kolibri-qsens allocate -base B.gguf -donors ... -sens sens.jsonl -budget 33 -out plan.json [-types-out t.txt]
//	kolibri-qsens build    -base B.gguf -donors ... -plan plan.json -out X.gguf
//
// measure scores the base file once and then, for each item and each donor
// type the item does not already have, a file that differs from the base in
// that item alone (gguf.Splice; nothing is requantized). Each record is
// appended to -out as a JSON line, so an interrupted run resumes where it
// stopped. The candidate file lives in -work and is deleted after its run.
//
// No command writes over an existing file: outputs are written under a
// temporary name and linked to their path, which must not exist.
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"syscall"
	"time"

	"kolibri-llm/internal/gguf"
	"kolibri-llm/internal/quantsens"
)

const gib = 1 << 30

func main() {
	log.SetFlags(0)
	if len(os.Args) < 2 {
		usage()
	}
	cmd, args := os.Args[1], os.Args[2:]
	var err error
	switch cmd {
	case "splice":
		err = cmdSplice(args, os.Stdout)
	case "measure":
		err = cmdMeasure(args, os.Stdout)
	case "allocate":
		err = cmdAllocate(args, os.Stdout)
	case "build":
		err = cmdBuild(args, os.Stdout)
	default:
		usage()
	}
	if err != nil {
		log.Fatalf("kolibri-qsens %s: %v", cmd, err)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: kolibri-qsens splice|measure|allocate|build [flags]; -h for a command's flags")
	os.Exit(2)
}

type multi []string

func (m *multi) String() string     { return strings.Join(*m, " ") }
func (m *multi) Set(s string) error { *m = append(*m, s); return nil }

// opened is a GGUF header with its file, for Splice. The files stay open
// until the process exits.
type opened struct {
	path string
	src  gguf.Source
}

func open(path string) (*opened, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	f, err := gguf.Read(bufio.NewReaderSize(fh, 1<<20))
	if err != nil {
		fh.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &opened{path: path, src: gguf.Source{File: f, Data: fh}}, nil
}

func openDonors(list string) ([]*opened, error) {
	var ds []*opened
	for p := range strings.SplitSeq(list, ",") {
		if p = strings.TrimSpace(p); p == "" {
			continue
		}
		d, err := open(p)
		if err != nil {
			return nil, err
		}
		ds = append(ds, d)
	}
	return ds, nil
}

// writeNew writes a file through fn under a temporary name in path's
// directory and links it to path, which must not exist.
func writeNew(path string, fn func(w io.Writer) error) error {
	if _, err := os.Lstat(path); err == nil {
		return fmt.Errorf("%s exists; refusing to overwrite it", path)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	bw := bufio.NewWriterSize(tmp, 8<<20)
	if err := fn(bw); err != nil {
		tmp.Close()
		return err
	}
	if err := bw.Flush(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Link(tmp.Name(), path)
}

func splice(base *opened, take map[string]gguf.Source, out string) (int64, error) {
	var n int64
	err := writeNew(out, func(w io.Writer) error {
		var err error
		n, err = gguf.Splice(w, base.src, take)
		return err
	})
	return n, err
}

// freeBytes is the space available to an unprivileged user on dir's volume.
func freeBytes(dir string) (int64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		return 0, err
	}
	return int64(st.Bavail) * int64(st.Bsize), nil
}

func cmdSplice(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("splice", flag.ContinueOnError)
	base := fs.String("base", "", "base GGUF: metadata, tensor order and every tensor not taken")
	out := fs.String("out", "", "output GGUF; must not exist")
	var takes multi
	fs.Var(&takes, "take", "DONOR.gguf=REGEX: take every base tensor whose name matches REGEX from DONOR (repeatable; a tensor may match only one donor)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *base == "" || *out == "" || len(takes) == 0 {
		return errors.New("-base, -out and at least one -take are required")
	}
	b, err := open(*base)
	if err != nil {
		return err
	}
	take := map[string]gguf.Source{}
	from := map[string]string{}
	for _, tk := range takes {
		path, pat, ok := strings.Cut(tk, "=")
		if !ok {
			return fmt.Errorf("-take %q: want DONOR.gguf=REGEX", tk)
		}
		re, err := regexp.Compile(pat)
		if err != nil {
			return err
		}
		d, err := open(path)
		if err != nil {
			return err
		}
		n := 0
		for _, t := range b.src.File.Tensors {
			if !re.MatchString(t.Name) {
				continue
			}
			if prev, dup := from[t.Name]; dup && prev != path {
				return fmt.Errorf("%s matches both %s and %s", t.Name, prev, path)
			}
			take[t.Name], from[t.Name] = d.src, path
			n++
		}
		if n == 0 {
			return fmt.Errorf("-take %q matches no tensor of the base", tk)
		}
		fmt.Fprintf(stdout, "take %4d tensors from %s (%s)\n", n, path, pat)
	}
	n, err := splice(b, take, *out)
	if err != nil {
		return err
	}
	fmt.Fprintf(stdout, "wrote %s: %.3f GiB\n", *out, float64(n)/gib)
	return nil
}

// donorFor returns, per type, the first donor that holds every tensor of
// the item in that one type, and the item's bytes in it.
func donorFor(it quantsens.Item, donors []*opened) (map[string]*opened, map[string]int64) {
	by, bytes := map[string]*opened{}, map[string]int64{}
	for _, d := range donors {
		n, typ, err := quantsens.ItemBytes(d.src.File, it)
		if err != nil || typ == "mixed" {
			continue
		}
		if _, ok := by[typ]; !ok {
			by[typ], bytes[typ] = d, n
		}
	}
	return by, bytes
}

func readRecords(path string) ([]quantsens.Record, error) {
	fh, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	var recs []quantsens.Record
	sc := bufio.NewScanner(fh)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for line := 1; sc.Scan(); line++ {
		if strings.TrimSpace(sc.Text()) == "" {
			continue
		}
		var r quantsens.Record
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		recs = append(recs, r)
	}
	return recs, sc.Err()
}

func appendRecord(path string, r quantsens.Record) error {
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	fh, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
	if err != nil {
		return err
	}
	if _, err := fh.Write(append(b, '\n')); err != nil {
		fh.Close()
		return err
	}
	return fh.Close()
}

// filterSet is a comma-separated list as a set, or nil (no filter) when
// empty. Type names compare in upper case.
func filterSet(list string, upper bool) map[string]bool {
	if list == "" {
		return nil
	}
	m := map[string]bool{}
	for s := range strings.SplitSeq(list, ",") {
		s = strings.TrimSpace(s)
		if upper {
			s = strings.ToUpper(s)
		}
		m[s] = true
	}
	return m
}

// scorer runs llama-perplexity --kl-divergence on a model and keeps its log.
type scorer struct {
	bin, kldBase, logDir string
	extra                []string
}

func (s scorer) score(model, logName string) (quantsens.Score, float64, string, error) {
	logPath := filepath.Join(s.logDir, logName)
	args := append([]string{"-m", model}, s.extra...)
	args = append(args, "--kl-divergence-base", s.kldBase, "--kl-divergence")
	t0 := time.Now()
	out, runErr := exec.Command(s.bin, args...).CombinedOutput()
	secs := time.Since(t0).Seconds()
	header := fmt.Sprintf("$ %s %s\n", s.bin, strings.Join(args, " "))
	if err := os.WriteFile(logPath, append([]byte(header), out...), 0o644); err != nil {
		return quantsens.Score{}, secs, logPath, err
	}
	if runErr != nil {
		return quantsens.Score{}, secs, logPath, fmt.Errorf("%s failed (%v); log %s", s.bin, runErr, logPath)
	}
	sc, err := quantsens.ParseKLD(string(out))
	if err != nil {
		return sc, secs, logPath, fmt.Errorf("%w; log %s", err, logPath)
	}
	return sc, secs, logPath, nil
}

func cmdMeasure(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("measure", flag.ContinueOnError)
	base := fs.String("base", "", "base GGUF")
	donors := fs.String("donors", "", "comma-separated donor GGUFs")
	scope := fs.String("scope", "kind", "kind: one item per tensor kind over all layers; layer: per kind and layer")
	kinds := fs.String("kinds", "", "measure only these kinds (comma-separated; default all)")
	types := fs.String("types", "", "measure only these types (comma-separated; default every donor type)")
	ppl := fs.String("perplexity", "", "llama-perplexity binary")
	kldBase := fs.String("kld-base", "", "llama-perplexity --kl-divergence-base file of the BF16 logits")
	pplArgs := fs.String("args", "-ngl 99 -t 10 --fit off -c 512 -b 512", "further llama-perplexity arguments (space-separated)")
	work := fs.String("work", "", "directory for the candidate file and the run logs (needs room for one candidate)")
	out := fs.String("out", "", "JSONL of the records; appended to, and read to resume")
	dry := fs.Bool("dry", false, "list the runs and their sizes, run nothing")
	limit := fs.Int("limit", 0, "stop after this many runs (0: no limit)")
	minFree := fs.Float64("min-free", 8, "GiB that must stay free beside a candidate")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *base == "" || *out == "" || (!*dry && (*ppl == "" || *kldBase == "" || *work == "")) {
		return errors.New("-base and -out are required, and -perplexity, -kld-base and -work unless -dry")
	}
	b, err := open(*base)
	if err != nil {
		return err
	}
	ds, err := openDonors(*donors)
	if err != nil {
		return err
	}
	items, err := quantsens.Items(b.src.File, *scope)
	if err != nil {
		return err
	}
	recs, err := readRecords(*out)
	if err != nil {
		return err
	}
	done := map[[3]string]bool{}
	var baseRec *quantsens.Record
	for i, r := range recs {
		done[[3]string{r.Scope, r.Item, r.Type}] = true
		if r.Scope == "base" {
			baseRec = &recs[i]
		}
	}
	kindSet, typeSet := filterSet(*kinds, false), filterSet(*types, true)
	st, err := os.Stat(*base)
	if err != nil {
		return err
	}

	type run struct {
		it               quantsens.Item
		typ, baseType    string
		donor            *opened
		bytes, baseBytes int64
		size             int64 // the candidate file's size
	}
	var runs []run
	for _, it := range items {
		if kindSet != nil && !kindSet[it.Kind] {
			continue
		}
		bb, bt, err := quantsens.ItemBytes(b.src.File, it)
		if err != nil {
			return err
		}
		by, bytes := donorFor(it, ds)
		for _, typ := range slices.Sorted(maps.Keys(by)) {
			if typ == bt || (typeSet != nil && !typeSet[typ]) || done[[3]string{*scope, it.Name, typ}] {
				continue
			}
			runs = append(runs, run{it: it, typ: typ, baseType: bt, donor: by[typ], bytes: bytes[typ], baseBytes: bb, size: st.Size() - bb + bytes[typ]})
		}
	}
	fmt.Fprintf(stdout, "base %s: %d %s items, %d runs to do", *base, len(items), *scope, len(runs))
	if baseRec == nil {
		fmt.Fprint(stdout, " plus the base itself")
	}
	fmt.Fprintln(stdout)
	if *dry {
		for _, r := range runs {
			fmt.Fprintf(stdout, "  %-24s %-8s -> %-8s %8.3f -> %8.3f GiB, file %.2f GiB (%s)\n", r.it.Name, r.baseType, r.typ,
				float64(r.baseBytes)/gib, float64(r.bytes)/gib, float64(r.size)/gib, r.donor.path)
		}
		return nil
	}
	logDir := filepath.Join(*work, "logs")
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		return err
	}
	sc := scorer{bin: *ppl, kldBase: *kldBase, logDir: logDir, extra: strings.Fields(*pplArgs)}
	if baseRec == nil {
		s, secs, lp, err := sc.score(*base, "base.log")
		if err != nil {
			return err
		}
		r := quantsens.Record{Scope: "base", KLD: s.KLD, KLDErr: s.KLDErr, SameTop: s.SameTop, LnPPLRatio: s.LnPPLRatio, Seconds: secs, Log: lp}
		if err := appendRecord(*out, r); err != nil {
			return err
		}
		baseRec = &r
		fmt.Fprintf(stdout, "base: KLD %.6f ± %.6f, same top %.3f%% (%.0f s)\n", s.KLD, s.KLDErr, s.SameTop, secs)
	}
	cand := filepath.Join(*work, "candidate.gguf")
	for i, r := range runs {
		if *limit > 0 && i >= *limit {
			fmt.Fprintf(stdout, "stopped after %d runs (-limit)\n", i)
			break
		}
		free, err := freeBytes(*work)
		if err != nil {
			return err
		}
		if need := r.size + int64(*minFree*gib); free < need {
			return fmt.Errorf("%.1f GiB free in %s; a candidate needs %.1f GiB plus %.0f GiB to spare", float64(free)/gib, *work, float64(r.size)/gib, *minFree)
		}
		take := map[string]gguf.Source{}
		for _, name := range r.it.Tensors {
			take[name] = r.donor.src
		}
		if _, err := splice(b, take, cand); err != nil {
			return err
		}
		s, secs, lp, err := sc.score(cand, fmt.Sprintf("%s-%s-%s.log", *scope, r.it.Name, r.typ))
		if rmErr := os.Remove(cand); rmErr != nil && err == nil {
			err = rmErr
		}
		if err != nil {
			return err
		}
		rec := quantsens.Record{Scope: *scope, Item: r.it.Name, Type: r.typ, Bytes: r.bytes, BaseType: r.baseType, BaseBytes: r.baseBytes,
			KLD: s.KLD, KLDErr: s.KLDErr, SameTop: s.SameTop, LnPPLRatio: s.LnPPLRatio, DKL: s.KLD - baseRec.KLD, Seconds: secs, Log: lp}
		if err := appendRecord(*out, rec); err != nil {
			return err
		}
		fmt.Fprintf(stdout, "[%d/%d] %-24s %-8s -> %-8s KLD %.6f (Δ %+.6f), same top %.3f%% (%.0f s)\n", i+1, len(runs), r.it.Name, r.baseType, r.typ,
			s.KLD, rec.DKL, s.SameTop, secs)
	}
	return nil
}

// infos describes every layer-scope item of the base for BuildChoices, and
// returns the bytes of the tensors no item covers.
func infos(b *opened, ds []*opened) ([]quantsens.Item, []quantsens.ItemInfo, int64, error) {
	items, err := quantsens.Items(b.src.File, "layer")
	if err != nil {
		return nil, nil, 0, err
	}
	covered := map[string]bool{}
	var out []quantsens.ItemInfo
	for _, it := range items {
		bb, bt, err := quantsens.ItemBytes(b.src.File, it)
		if err != nil {
			return nil, nil, 0, err
		}
		_, bytes := donorFor(it, ds)
		out = append(out, quantsens.ItemInfo{Item: it, BaseType: bt, BaseBytes: bb, Donor: bytes})
		for _, n := range it.Tensors {
			covered[n] = true
		}
	}
	var fixed int64
	for _, t := range b.src.File.Tensors {
		if covered[t.Name] {
			continue
		}
		n, err := t.Bytes()
		if err != nil {
			return nil, nil, 0, err
		}
		fixed += n
	}
	return items, out, fixed, nil
}

func cmdAllocate(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("allocate", flag.ContinueOnError)
	base := fs.String("base", "", "base GGUF the measurements were taken against")
	donors := fs.String("donors", "", "comma-separated donor GGUFs (for each type's bytes)")
	sens := fs.String("sens", "", "measure's JSONL")
	budget := fs.Float64("budget", 0, "tensor bytes allowed, GiB")
	unit := fs.Int64("unit", 256, "knapsack granularity, KiB (sizes round up)")
	out := fs.String("out", "", "plan JSON; must not exist")
	typesOut := fs.String("types-out", "", "also write a llama-quantize --tensor-type-file; must not exist")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *base == "" || *sens == "" || *budget <= 0 || *out == "" {
		return errors.New("-base, -sens, -budget and -out are required")
	}
	b, err := open(*base)
	if err != nil {
		return err
	}
	ds, err := openDonors(*donors)
	if err != nil {
		return err
	}
	recs, err := readRecords(*sens)
	if err != nil {
		return err
	}
	items, in, fixed, err := infos(b, ds)
	if err != nil {
		return err
	}
	choices := quantsens.BuildChoices(in, recs)
	p, err := quantsens.Allocate(choices, fixed, int64(*budget*gib), *unit*1024)
	if err != nil {
		return err
	}
	if err := writeNew(*out, func(w io.Writer) error {
		enc := json.NewEncoder(w)
		enc.SetIndent("", " ")
		return enc.Encode(p)
	}); err != nil {
		return err
	}
	if *typesOut != "" {
		s, err := quantsens.TensorTypes(p, items)
		if err != nil {
			return err
		}
		if err := writeNew(*typesOut, func(w io.Writer) error { _, err := io.WriteString(w, s); return err }); err != nil {
			return err
		}
	}
	summary(stdout, p, choices)
	return nil
}

// summary prints the plan's size, predicted added KL, and how many items of
// each kind got each type.
func summary(w io.Writer, p quantsens.Plan, choices []quantsens.Choice) {
	fmt.Fprintf(w, "plan: %.3f GiB of %.3f GiB (fixed %.3f), predicted ΔKL %+.6f over the base\n",
		float64(p.Total)/gib, float64(p.Budget)/gib, float64(p.Fixed)/gib, p.DKL)
	count := map[string]map[string]int{}
	derived := 0
	for _, pk := range p.Picks {
		kind := pk.Item[strings.LastIndex(pk.Item, ".")+1:] // blk.12.attn -> attn; output -> output
		if count[kind] == nil {
			count[kind] = map[string]int{}
		}
		count[kind][pk.Type]++
		if pk.Derived {
			derived++
		}
	}
	for _, k := range quantsens.Kinds {
		if count[k] == nil {
			continue
		}
		var parts []string
		for _, t := range slices.Sorted(maps.Keys(count[k])) {
			parts = append(parts, fmt.Sprintf("%s ×%d", t, count[k][t]))
		}
		fmt.Fprintf(w, "  %-14s %s\n", k, strings.Join(parts, ", "))
	}
	opts := 0
	for _, c := range choices {
		opts += len(c.Options)
	}
	fmt.Fprintf(w, "  %d items, %d options; %d picks use a cost spread from a kind-scope measurement\n", len(choices), opts, derived)
}

func cmdBuild(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("build", flag.ContinueOnError)
	base := fs.String("base", "", "base GGUF the plan was allocated against")
	donors := fs.String("donors", "", "comma-separated donor GGUFs")
	planPath := fs.String("plan", "", "allocate's plan JSON")
	out := fs.String("out", "", "output GGUF; must not exist")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *base == "" || *planPath == "" || *out == "" {
		return errors.New("-base, -plan and -out are required")
	}
	b, err := open(*base)
	if err != nil {
		return err
	}
	ds, err := openDonors(*donors)
	if err != nil {
		return err
	}
	raw, err := os.ReadFile(*planPath)
	if err != nil {
		return err
	}
	var p quantsens.Plan
	if err := json.Unmarshal(raw, &p); err != nil {
		return err
	}
	items, err := quantsens.Items(b.src.File, "layer")
	if err != nil {
		return err
	}
	byName := map[string]quantsens.Item{}
	for _, it := range items {
		byName[it.Name] = it
	}
	take := map[string]gguf.Source{}
	want := map[string]string{}
	for _, pk := range p.Picks {
		it, ok := byName[pk.Item]
		if !ok {
			return fmt.Errorf("plan item %s is not an item of %s", pk.Item, *base)
		}
		_, bt, err := quantsens.ItemBytes(b.src.File, it)
		if err != nil {
			return err
		}
		for _, n := range it.Tensors {
			want[n] = pk.Type
		}
		if pk.Type == bt {
			continue
		}
		by, _ := donorFor(it, ds)
		d, ok := by[pk.Type]
		if !ok {
			return fmt.Errorf("%s: no donor holds it in %s", pk.Item, pk.Type)
		}
		for _, n := range it.Tensors {
			take[n] = d.src
		}
	}
	n, err := splice(b, take, *out)
	if err != nil {
		return err
	}
	got, err := gguf.Open(*out)
	if err != nil {
		return err
	}
	var off []string
	for _, t := range got.Tensors {
		if w, ok := want[t.Name]; ok && t.Type.String() != w {
			off = append(off, fmt.Sprintf("%s %s (want %s)", t.Name, t.Type, w))
		}
	}
	if len(off) > 0 {
		return fmt.Errorf("%s: %d tensors not as planned, e.g. %v", *out, len(off), off[:min(4, len(off))])
	}
	fmt.Fprintf(stdout, "wrote %s: %.3f GiB, %d tensors taken from donors, every planned tensor as planned\n", *out, float64(n)/gib, len(take))
	return nil
}
