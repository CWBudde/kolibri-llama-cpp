// Command kolibri-loops is the reproducible baseline of Kolibri-1's repetition
// loops (issue #66, step 1; docs/loop-baseline.md).
//
//	kolibri-loops run -out DIR -server BIN -model GGUF [flags] [-- server flags]
//	kolibri-loops report -out DIR [-json FILE]
//
// run drives one llama-server through testdata/loops/corpus.jsonl, one prompt
// at a time, greedy with fixed settings and a seed, and writes one record per
// prompt. It starts the server itself, or attaches to one with -url. A second
// run with the same -out continues where the first stopped, and refuses if the
// weight, the server binary, the chat template, the corpus or a setting moved.
//
// report recomputes every number from the records alone. No weight is edited,
// and no timing is taken.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

func expandHome(p string) string {
	if rest, ok := strings.CutPrefix(p, "~/"); ok {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, rest)
		}
	}
	return p
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("kolibri-loops: ")
	if len(os.Args) < 2 {
		log.Fatal("usage: kolibri-loops run|report [flags]")
	}
	var err error
	switch os.Args[1] {
	case "run":
		err = runMain(os.Args[2:])
	case "report":
		err = reportMain(os.Args[2:])
	default:
		err = fmt.Errorf("unknown mode %q: use run or report", os.Args[1])
	}
	if err != nil {
		log.Fatal(err)
	}
}

func runMain(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	cfg := runConfig{Logf: log.Printf, Settings: Settings{Rule: FixedRule}}
	fs.StringVar(&cfg.CorpusPath, "corpus", "testdata/loops/corpus.jsonl", "the corpus")
	fs.StringVar(&cfg.Out, "out", "", "results directory, resumable")
	fs.StringVar(&cfg.Model, "model", "", "the weight; hashed, and checked against the server's")
	fs.StringVar(&cfg.Server, "server", "", "llama-server; started unless -url is set, always hashed")
	fs.StringVar(&cfg.URL, "url", "", "a running llama-server, instead of starting -server")
	fs.IntVar(&cfg.Port, "port", 0, "port of a started server; 0 picks a free one")
	fs.IntVar(&cfg.Settings.NPredict, "n-predict", defaultNPredict, "tokens per prompt")
	fs.IntVar(&cfg.Settings.NPredictThink, "n-predict-think", defaultNPredictThink, "tokens per prompt with thinking on")
	fs.IntVar(&cfg.Settings.Seed, "seed", defaultSeed, "request seed")
	fs.IntVar(&cfg.Settings.NProbs, "n-probs", defaultNProbs, "candidates kept per generated token")
	fs.IntVar(&cfg.Limit, "limit", 0, "stop after this many new records; 0 runs all")
	fs.DurationVar(&cfg.LoadTimeout, "load-timeout", 30*time.Minute, "how long the server may take to load")
	fs.DurationVar(&cfg.RequestTimeout, "request-timeout", 60*time.Minute, "how long one request may take")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if cfg.Out == "" || cfg.Model == "" || cfg.Server == "" {
		fs.Usage()
		return fmt.Errorf("-out, -model and -server are required")
	}
	cfg.Out, cfg.Model, cfg.Server = expandHome(cfg.Out), expandHome(cfg.Model), expandHome(cfg.Server)
	cfg.ServerArgs = fs.Args()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	return run(ctx, cfg)
}

func reportMain(args []string) error {
	fs := flag.NewFlagSet("report", flag.ExitOnError)
	out := fs.String("out", "", "results directory")
	jsonPath := fs.String("json", "", "also write the report as JSON to this file")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *out == "" || fs.NArg() != 0 {
		fs.Usage()
		return fmt.Errorf("-out is required")
	}
	rs, err := readRecords(expandHome(*out))
	if err != nil {
		return err
	}
	rep, err := buildReport(rs)
	if err != nil {
		return err
	}
	writeReport(os.Stdout, rep)
	if *jsonPath != "" {
		data, err := json.MarshalIndent(rep, "", " ")
		if err != nil {
			return err
		}
		return os.WriteFile(expandHome(*jsonPath), append(data, '\n'), 0o644)
	}
	return nil
}
