// Command kolibri-bridge serves an OpenAI-compatible chat endpoint in front of llama-server and runs
// the deterministic operations a model asks for through tool calls: internal/bridge, issue #59. It
// is runtime tool execution, not hidden-state integration. See docs/execution-bridge.md.
//
//	llama-server -m MODEL --jinja --port 8099 ...
//	kolibri-bridge -upstream http://127.0.0.1:8099 -listen 127.0.0.1:8098 -trace trace.jsonl
//	curl 127.0.0.1:8098/v1/chat/completions -d '{"messages":[{"role":"user","content":"..."}]}'
//
// -ops prints the protocol version, the limits and each operation's schemas, and exits.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"kolibri-llm/internal/bridge"
	"kolibri-llm/internal/bridge/ops"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, os.Args[1:], os.Stdout, os.Stderr, nil); err != nil {
		if !errors.Is(err, flag.ErrHelp) {
			log.Print(err)
		}
		os.Exit(1)
	}
}

// run is main without the process: ready, if set, gets the address once the server listens.
func run(ctx context.Context, args []string, stdout, stderr io.Writer, ready func(addr string)) error {
	lim := bridge.DefaultLimits()
	opLim := ops.DefaultLimits()
	fs := flag.NewFlagSet("kolibri-bridge", flag.ContinueOnError)
	fs.SetOutput(stderr)
	listen := fs.String("listen", "127.0.0.1:8098", "address to serve on; the API has no authentication, keep it local")
	upstream := fs.String("upstream", "http://127.0.0.1:8099", "llama-server's base URL")
	allow := fs.String("allow", "", "comma-separated operations to allow; empty allows all")
	tracePath := fs.String("trace", "", "append every trace event to this file, as JSON lines")
	marker := fs.String("call-marker", "", `text that marks an unparsed call in an answer; empty means "<tool_call>"`)
	listOps := fs.Bool("ops", false, "print the protocol version, the limits and the operations, and exit")
	fs.IntVar(&lim.MaxTurns, "max-turns", lim.MaxTurns, "completions per request")
	fs.IntVar(&lim.MaxCalls, "max-calls", lim.MaxCalls, "operations per request")
	fs.IntVar(&lim.MaxCallsPerTurn, "max-calls-per-turn", lim.MaxCallsPerTurn, "operations per completion")
	fs.IntVar(&lim.MaxRequests, "max-requests", lim.MaxRequests, "open requests")
	fs.IntVar(&lim.MaxRunningOps, "max-running-ops", lim.MaxRunningOps, "operations running at once")
	fs.IntVar(&lim.MaxTurnTokens, "max-turn-tokens", lim.MaxTurnTokens, "max_tokens of a completion")
	fs.Int64Var(&lim.MaxTurnBytes, "max-turn-bytes", lim.MaxTurnBytes, "bytes of a completion's response")
	fs.Int64Var(&lim.MaxRequestBytes, "max-request-bytes", lim.MaxRequestBytes, "bytes of a client's request")
	fs.DurationVar(&lim.OpTimeout, "op-timeout", lim.OpTimeout, "one operation's run time")
	fs.DurationVar(&lim.TurnTimeout, "turn-timeout", lim.TurnTimeout, "one completion, prompt processing included")
	fs.DurationVar(&lim.RequestTimeout, "request-timeout", lim.RequestTimeout, "the whole request")
	fs.IntVar(&opLim.MaxArgBytes, "max-arg-bytes", opLim.MaxArgBytes, "bytes of an operation's arguments")
	fs.IntVar(&opLim.MaxIntBits, "max-int-bits", opLim.MaxIntBits, "bits of an integer input or result")
	fs.IntVar(&opLim.MaxArrayLen, "max-array-len", opLim.MaxArrayLen, "elements of an array input")
	fs.IntVar(&opLim.MaxResultBytes, "max-result-bytes", opLim.MaxResultBytes, "bytes of an encoded result")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected arguments: %q", fs.Args())
	}

	cfg := bridge.Config{Upstream: *upstream, Limits: lim, OpLimits: opLim, CallMarker: *marker}
	if *allow != "" {
		cfg.Allow = strings.Split(*allow, ",")
	}
	if *tracePath != "" {
		file, err := os.OpenFile(*tracePath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		defer file.Close()
		var mu sync.Mutex
		enc := json.NewEncoder(file)
		cfg.Trace = func(e bridge.Event) {
			mu.Lock()
			defer mu.Unlock()
			if err := enc.Encode(e); err != nil {
				log.Printf("trace: %v", err)
			}
		}
	}
	b, err := bridge.New(cfg)
	if err != nil {
		return err
	}
	closeBridge := func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := b.Close(ctx); err != nil {
			log.Printf("close: %v", err)
		}
	}
	if *listOps {
		defer closeBridge()
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(b.Manifest())
	}

	listener, err := net.Listen("tcp", *listen)
	if err != nil {
		closeBridge()
		return err
	}
	// No WriteTimeout: a response lasts as long as its request, which has limits of its own.
	srv := &http.Server{Handler: b.Handler(), ReadHeaderTimeout: 10 * time.Second}
	served := make(chan error, 1)
	go func() { served <- srv.Serve(listener) }()
	log.Printf("kolibri-bridge on %s, upstream %s, protocol %d", listener.Addr(), *upstream, ops.Version)
	if ready != nil {
		ready(listener.Addr().String())
	}
	select {
	case err := <-served:
		closeBridge()
		return err
	case <-ctx.Done():
	}
	// Close the bridge first: it ends the running requests, so that Shutdown has nothing to wait for.
	closeBridge()
	shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return srv.Shutdown(shutdown)
}
