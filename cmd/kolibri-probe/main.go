// Command kolibri-probe generates and verifies the prompts of the layer-depth probe of issue #63
// and reports linear probes over captured activations: docs/layer-probe.md.
//
//	kolibri-probe generate [-seed N] [-out testdata/probe/smoke.jsonl]
//	kolibri-probe verify   [-dataset testdata/probe/smoke.jsonl]
//	kolibri-probe report   -capture DIR [-dataset ...] [-folds 5] [-l2 0.01] [-shuffles 20] [-out DIR]
//
// tools/probe/capture.py writes the capture directory. report refuses a capture whose dataset,
// activations or shape do not match its capture.json, or that was not checked: see ReadCapture. It
// never replaces a report. Flags that move a committed definition write report-deviation-*, which
// says so at its top; only the committed definitions write report.md.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"kolibri-llm/internal/probe"
)

const defaultDataset = "testdata/probe/smoke.jsonl"

func usage() {
	fmt.Fprintln(os.Stderr, "usage: kolibri-probe generate|verify|report [flags]")
	os.Exit(2)
}

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	var err error
	switch os.Args[1] {
	case "generate":
		err = generate(os.Args[2:])
	case "verify":
		err = verify(os.Args[2:])
	case "report":
		err = report(os.Args[2:])
	default:
		usage()
	}
	if err != nil {
		log.Fatalf("FAIL %v", err)
	}
}

func generate(args []string) error {
	fs := flag.NewFlagSet("generate", flag.ExitOnError)
	seed := fs.Uint64("seed", probe.Seed, "the generator's seed")
	out := fs.String("out", defaultDataset, "output JSONL")
	if err := fs.Parse(args); err != nil {
		return err
	}
	items := probe.Generate(*seed)
	if err := check(items); err != nil {
		return err
	}
	data, err := probe.MarshalJSONL(items)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(*out, data, 0o644); err != nil {
		return err
	}
	fmt.Printf("PASS wrote %d prompts to %s\n", len(items), *out)
	return nil
}

func check(items []probe.Item) error {
	errs := append(probe.VerifyAnswers(items), probe.VerifyStructure(items)...)
	for _, e := range errs {
		fmt.Fprintln(os.Stderr, e)
	}
	if len(errs) > 0 {
		return fmt.Errorf("%d problems in the dataset", len(errs))
	}
	return nil
}

func readDataset(path string) ([]probe.Item, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return probe.ReadJSONL(f)
}

func verify(args []string) error {
	fs := flag.NewFlagSet("verify", flag.ExitOnError)
	path := fs.String("dataset", defaultDataset, "dataset JSONL")
	if err := fs.Parse(args); err != nil {
		return err
	}
	items, err := readDataset(*path)
	if err != nil {
		return err
	}
	if err := check(items); err != nil {
		return err
	}
	fmt.Printf("PASS %d prompts: the oracle reads every operation and computes every answer from the prompt, each refusal carries its partner's numbers, and no template, seed or sentence frame spans two groups\n", len(items))
	return nil
}
