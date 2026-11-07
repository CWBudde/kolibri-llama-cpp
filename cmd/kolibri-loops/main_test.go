package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// corpusSHA256 pins testdata/loops/corpus.jsonl and with it the split: a later
// edit has to change this line in plain sight.
const corpusSHA256 = "722992046f85c1ac63394e954dd776239d08ad7a283b4117cdd1140bf36b2ddc"

const realCorpus = "../../testdata/loops/corpus.jsonl"

func TestMain(m *testing.M) {
	if os.Getenv("KOLIBRI_LOOPS_FAKE_ARGS") != "" {
		fakeServerMain()
		return
	}
	os.Exit(m.Run())
}

// fixture is a small valid corpus, one weight file and one server binary.
type fixture struct {
	corpus, weight, server, out string
}

func fixtureItems() []Item {
	berlin := &Answer{Accept: []string{"1001"}} // the fake's second generated token
	var items []Item
	add := func(pair, stratum string, cells ...[3]string) {
		for _, c := range cells {
			it := Item{ID: pair + "." + c[0] + "." + c[1], Pair: pair, Group: pair, Lang: c[0], Form: c[1], Stratum: stratum, Prompt: c[2]}
			if stratum == stratumAnswer {
				it.Answer = berlin
			}
			items = append(items, it)
		}
	}
	add("a", stratumAnswer, [3]string{"de", formRaw, "Die Hauptstadt von Deutschland ist"}, [3]string{"en", formRaw, "The capital of Germany is"}, [3]string{"de", formChat, "Wie?"}, [3]string{"en", formChat, "How?"})
	add("b", stratumOpen, [3]string{"de", formRaw, "Es war"}, [3]string{"en", formRaw, "Once"}, [3]string{"de", formChat, "Warum?"}, [3]string{"en", formChat, "Why?"})
	add("t", stratumThink, [3]string{"de", formChatThink, "Denk?"}, [3]string{"en", formChatThink, "Think?"})
	add("x", stratumText, [3]string{"de", formText, "Ein Text."}, [3]string{"en", formText, "A text."})
	splits := assignSplits(items)
	for i := range items {
		items[i].Split = splits[items[i].Group]
	}
	return items
}

func writeCorpus(t *testing.T, path string, items []Item) {
	t.Helper()
	var b bytes.Buffer
	for _, it := range items {
		data, err := json.Marshal(it)
		if err != nil {
			t.Fatal(err)
		}
		b.Write(append(data, '\n'))
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func newFixture(t *testing.T) fixture {
	t.Helper()
	dir := t.TempDir()
	f := fixture{corpus: filepath.Join(dir, "corpus.jsonl"), weight: filepath.Join(dir, "w.gguf"), server: filepath.Join(dir, "llama-server"), out: filepath.Join(dir, "out")}
	writeCorpus(t, f.corpus, fixtureItems())
	for _, p := range []string{f.weight, f.server, f.lib()} {
		if err := os.WriteFile(p, []byte("bytes of "+filepath.Base(p)), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	// a build's versioned link is not a second library
	if err := os.Symlink(f.lib(), filepath.Join(dir, "libllama-server.0.dylib")); err != nil {
		t.Fatal(err)
	}
	return f
}

// lib is the library that holds the fixture server's code, as a real build's
// libllama-server-impl does.
func (f fixture) lib() string {
	return filepath.Join(filepath.Dir(f.server), "libllama-server-impl.dylib")
}

func (f fixture) config(url string) runConfig {
	return runConfig{
		CorpusPath: f.corpus, Out: f.out, Model: f.weight, Server: f.server, URL: url, AttachedResume: url != "",
		Settings:    Settings{NPredict: 32, NPredictThink: 48, Seed: 42, NProbs: 2, Rule: FixedRule},
		LoadTimeout: 30 * time.Second, RequestTimeout: 30 * time.Second, Logf: func(string, ...any) {},
	}
}

func recordNames(t *testing.T, out string) []string {
	t.Helper()
	entries, err := os.ReadDir(recordsDir(out))
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	return names
}

func TestRecordHoldsThePins(t *testing.T) {
	f := newFixture(t)
	fake := newFake(f.weight)
	if err := run(context.Background(), f.config(startFake(t, fake))); err != nil {
		t.Fatal(err)
	}
	r, err := readRecord(recordPath(f.out, "a.de.raw"))
	if err != nil {
		t.Fatal(err)
	}
	real, _ := filepath.EvalSymlinks(f.weight)
	weightHash, _ := hashFile(f.weight)
	serverHash, _ := hashFile(f.server)
	if r.Pins.WeightRealpath != real || r.Pins.WeightSHA256 != weightHash || r.Pins.ServerSHA256 != serverHash {
		t.Fatalf("pins = %+v", r.Pins)
	}
	if r.Pins.ChatTemplate != fakeTemplate || r.Pins.ChatTemplateSHA256 != sha256String(fakeTemplate) || r.Pins.ServerBuildInfo != "fake-1" {
		t.Fatalf("server pins = %+v", r.Pins)
	}
	if !slices.Equal(r.PromptTokens, documentedTokens["Die Hauptstadt von Deutschland ist"]) || len(r.GeneratedTokens) != 32 {
		t.Fatalf("tokens: prompt %v, generated %d", r.PromptTokens, len(r.GeneratedTokens))
	}
	if !slices.ContainsFunc(fake.bodies, func(b []byte) bool { return bytes.Equal(b, r.Request) }) {
		t.Fatalf("the record's request is not the bytes sent: %s", r.Request)
	}
	var req completionRequest
	if err := json.Unmarshal(r.Request, &req); err != nil {
		t.Fatal(err)
	}
	if req.Temperature != 0 || req.Seed != 42 || req.CachePrompt || req.PostSamplingProbs || req.RepeatPenalty != 1 || !slices.Equal(req.Samplers, []string{"temperature"}) {
		t.Fatalf("request = %s", r.Request)
	}
	chat, err := readRecord(recordPath(f.out, "b.de.chat"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(chat.RenderedPrompt, "</think>\n\n") || !strings.Contains(string(chat.TemplateRequest), `"reasoning_effort":"none"`) {
		t.Fatalf("chat record: %q, %s", chat.RenderedPrompt, chat.TemplateRequest)
	}
	think, err := readRecord(recordPath(f.out, "t.en.chat-think"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.HasSuffix(think.RenderedPrompt, "</think>\n\n") || strings.Contains(string(think.TemplateRequest), "reasoning_effort") || len(think.GeneratedTokens) != 48 {
		t.Fatalf("thinking record: %q, %s, %d tokens", think.RenderedPrompt, think.TemplateRequest, len(think.GeneratedTokens))
	}
}

func TestResume(t *testing.T) {
	f := newFixture(t)
	fake := newFake(f.weight)
	url := startFake(t, fake)
	cfg := f.config(url)
	cfg.Limit = 3
	if err := run(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	if got := fake.count("/completion"); got != 3 {
		t.Fatalf("%d completions after the first run; want 3", got)
	}
	first := recordNames(t, f.out)
	if len(first) != 3 {
		t.Fatalf("records %v", first)
	}
	before, err := os.ReadFile(filepath.Join(recordsDir(f.out), first[0]))
	if err != nil {
		t.Fatal(err)
	}
	// the temp file of a write that was cut off is not a record
	if err := os.WriteFile(filepath.Join(recordsDir(f.out), ".tmp-12345"), []byte("{cut"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run(context.Background(), f.config(url)); err != nil {
		t.Fatal(err)
	}
	// 10 generated prompts, and "Ein Text." and "A text." have 8 and 6 positions
	if got := fake.count("/completion"); got != 10+8+6 {
		t.Fatalf("%d completions in all; the done prompts ran again, or some did not", got)
	}
	after, err := os.ReadFile(filepath.Join(recordsDir(f.out), first[0]))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("a resume changed a finished record")
	}
	// a third run has nothing to do
	n := fake.count("/completion")
	if err := run(context.Background(), f.config(url)); err != nil {
		t.Fatal(err)
	}
	if fake.count("/completion") != n {
		t.Fatal("a finished run sent completions")
	}
}

func TestResumeAfterServerFailure(t *testing.T) {
	f := newFixture(t)
	fake := newFake(f.weight)
	fake.failAfter = 2
	url := startFake(t, fake)
	if err := run(context.Background(), f.config(url)); err == nil || !strings.Contains(err.Error(), "the slot is gone") {
		t.Fatalf("run = %v", err)
	}
	if got := len(recordNames(t, f.out)); got != 2 {
		t.Fatalf("%d records after the failure; want 2", got)
	}
	fake.failAfter = 0
	if err := run(context.Background(), f.config(url)); err != nil {
		t.Fatal(err)
	}
	if got := len(recordNames(t, f.out)); got != len(fixtureItems()) {
		t.Fatalf("%d records; want all", got)
	}
}

func TestInterruptKeepsRecords(t *testing.T) {
	f := newFixture(t)
	fake := newFake(f.weight)
	url := startFake(t, fake)
	ctx, cancel := context.WithCancel(context.Background())
	fake.gen = func(p []int, n int) ([]int, string) {
		if fake.count("/completion") == 3 {
			cancel()
		}
		return defaultGen(p, n)
	}
	err := run(ctx, f.config(url))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("run = %v", err)
	}
	fake.gen = defaultGen
	if err := run(context.Background(), f.config(url)); err != nil {
		t.Fatal(err)
	}
	if got := len(recordNames(t, f.out)); got != len(fixtureItems()) {
		t.Fatalf("%d records; want all", got)
	}
}

// A resume refuses before it sends anything if the run's identity moved.
func TestResumeRefusesMismatch(t *testing.T) {
	for _, c := range []struct {
		name  string
		field string
		// before says whether the refusal comes before any request
		before bool
		change func(t *testing.T, f fixture, cfg *runConfig, fake *fakeServer)
	}{
		{"weight bytes", "weight_sha256", true, func(t *testing.T, f fixture, _ *runConfig, _ *fakeServer) {
			if err := os.WriteFile(f.weight, []byte("bytes of w.gguX"), 0o755); err != nil {
				t.Fatal(err)
			}
		}},
		{"server binary", "server_sha256", true, func(t *testing.T, f fixture, _ *runConfig, _ *fakeServer) {
			if err := os.WriteFile(f.server, []byte("another build"), 0o755); err != nil {
				t.Fatal(err)
			}
		}},
		// the binary is a stub: a rebuild that changes only the code in a library
		// leaves its hash and the commit alone
		{"a library of the server", "server_libs", true, func(t *testing.T, f fixture, _ *runConfig, _ *fakeServer) {
			if err := os.WriteFile(f.lib(), []byte("another build"), 0o755); err != nil {
				t.Fatal(err)
			}
		}},
		{"a library added next to the server", "server_libs", true, func(t *testing.T, f fixture, _ *runConfig, _ *fakeServer) {
			if err := os.WriteFile(filepath.Join(filepath.Dir(f.server), "libggml-metal.so"), []byte("x"), 0o755); err != nil {
				t.Fatal(err)
			}
		}},
		{"corpus", "corpus_sha256", true, func(t *testing.T, f fixture, _ *runConfig, _ *fakeServer) {
			items := fixtureItems()
			items[0].Prompt = "Die Hauptstadt"
			writeCorpus(t, f.corpus, items)
		}},
		{"a setting", "settings", true, func(_ *testing.T, _ fixture, cfg *runConfig, _ *fakeServer) {
			cfg.Settings.NPredict = 64
		}},
		{"the seed", "settings", true, func(_ *testing.T, _ fixture, cfg *runConfig, _ *fakeServer) {
			cfg.Settings.Seed = 43
		}},
		{"the detector", "settings", true, func(_ *testing.T, _ fixture, cfg *runConfig, _ *fakeServer) {
			cfg.Settings.Rule.MinRegion = 8
		}},
		{"chat template", "chat_template", false, func(_ *testing.T, _ fixture, _ *runConfig, fake *fakeServer) {
			fake.template = "{{ another }}"
		}},
		{"server build", "server_build_info", false, func(_ *testing.T, _ fixture, _ *runConfig, fake *fakeServer) {
			fake.buildInfo = "fake-2"
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := newFixture(t)
			fake := newFake(f.weight)
			cfg := f.config(startFake(t, fake))
			cfg.Limit = 2
			if err := run(context.Background(), cfg); err != nil {
				t.Fatal(err)
			}
			fake.requests = map[string]int{}
			cfg.Limit = 0
			c.change(t, f, &cfg, fake)
			err := run(context.Background(), cfg)
			if err == nil || !strings.Contains(err.Error(), "refusing to resume") || !strings.Contains(err.Error(), c.field) {
				t.Fatalf("run = %v; want a refusal that names %s", err, c.field)
			}
			if c.before && fake.total() != 0 {
				t.Fatalf("%d requests before the refusal", fake.total())
			}
			if got := fake.count("/completion"); got != 0 {
				t.Fatalf("%d completions after a refused resume", got)
			}
			if got := len(recordNames(t, f.out)); got != 2 {
				t.Fatalf("%d records after a refused resume", got)
			}
		})
	}
}

func TestServerLibsPinTheCode(t *testing.T) {
	f := newFixture(t)
	libs, err := serverLibs(f.server)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := hashFile(f.lib())
	if len(libs) != 1 || libs[filepath.Base(f.lib())] != want {
		t.Fatalf("libs = %v; want the one library once, not its link", libs)
	}
	if err := run(context.Background(), f.config(startFake(t, newFake(f.weight)))); err != nil {
		t.Fatal(err)
	}
	rs, err := readRecords(f.out)
	if err != nil {
		t.Fatal(err)
	}
	if rs[0].Pins.ServerLibs[filepath.Base(f.lib())] != want {
		t.Fatalf("pins = %+v", rs[0].Pins)
	}
}

// A link that leaves the directory is the only trace of that library's code, and
// a build that does not embed its Metal library reads it from next to the binary.
func TestServerLibsFollowTheCode(t *testing.T) {
	f := newFixture(t)
	dir := filepath.Dir(f.server)
	outside := filepath.Join(t.TempDir(), "libggml-metal.0.dylib")
	if err := os.WriteFile(outside, []byte("code outside"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, "libggml-metal.dylib")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "default.metallib"), []byte("shaders"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "llama-cli"), []byte("another tool"), 0o755); err != nil {
		t.Fatal(err)
	}
	libs, err := serverLibs(f.server)
	if err != nil {
		t.Fatal(err)
	}
	if len(libs) != 3 || libs["libggml-metal.dylib"] != sha256String("code outside") || libs["default.metallib"] != sha256String("shaders") {
		t.Fatalf("libs = %v", libs)
	}
	// the code behind the link moves, and the pin with it
	if err := os.WriteFile(outside, []byte("other code"), 0o755); err != nil {
		t.Fatal(err)
	}
	if libs, err = serverLibs(f.server); err != nil || libs["libggml-metal.dylib"] != sha256String("other code") {
		t.Fatalf("libs = %v, %v", libs, err)
	}
	if err := os.Remove(outside); err != nil {
		t.Fatal(err)
	}
	if _, err := serverLibs(f.server); err == nil || !strings.Contains(err.Error(), "libggml-metal.dylib") {
		t.Fatalf("a dangling library link: %v", err)
	}
}

// The server's API does not report its flags, so nothing could tell a restarted
// server on another ubatch from the first.
func TestAttachedServerDoesNotResume(t *testing.T) {
	f := newFixture(t)
	fake := newFake(f.weight)
	cfg := f.config(startFake(t, fake))
	cfg.Limit, cfg.AttachedResume = 1, false
	if err := run(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	fake.requests = map[string]int{}
	cfg.Limit = 0
	err := run(context.Background(), cfg)
	if err == nil || !strings.Contains(err.Error(), "refusing to resume") || !strings.Contains(err.Error(), "does not report its flags") {
		t.Fatalf("run = %v", err)
	}
	if fake.total() != 0 {
		t.Fatalf("%d requests before the refusal", fake.total())
	}
	cfg.AttachedResume = true
	if err := run(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	rs, err := readRecords(f.out)
	if err != nil {
		t.Fatal(err)
	}
	rep, err := buildReport(rs)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	writeReport(&out, rep)
	if !strings.Contains(out.String(), "server flags and environment: unknown") {
		t.Fatalf("an attached run's report does not say its flags are unknown:\n%s", out.String())
	}
}

// common/arg.cpp reads LLAMA_ARG_UBATCH, and a started server inherits it.
func TestResumeRefusesServerEnvironment(t *testing.T) {
	f := newFixture(t)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("KOLIBRI_LOOPS_FAKE_ARGS", filepath.Join(t.TempDir(), "args"))
	cfg := f.config("")
	cfg.Server, cfg.Limit = exe, 1
	if err := run(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLAMA_ARG_UBATCH", "21")
	err = run(context.Background(), cfg)
	if err == nil || !strings.Contains(err.Error(), "refusing to resume") || !strings.Contains(err.Error(), "server_env") {
		t.Fatalf("run = %v; want a refusal that names server_env", err)
	}
	if got := len(recordNames(t, f.out)); got != 1 {
		t.Fatalf("%d records after a refused resume", got)
	}
	g := newFixture(t)
	cfg = g.config("")
	cfg.Server, cfg.Limit = exe, 1
	t.Setenv("GGML_SOMETHING", "1")
	if err := run(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	r, err := readRecord(recordPath(g.out, "a.de.raw"))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(r.Pins.ServerEnv, []string{"GGML_SOMETHING=1", "LLAMA_ARG_UBATCH=21"}) {
		t.Fatalf("server_env = %v", r.Pins.ServerEnv)
	}
}

func TestResumeRefusesForeignRecord(t *testing.T) {
	f := newFixture(t)
	fake := newFake(f.weight)
	cfg := f.config(startFake(t, fake))
	cfg.Limit = 1
	if err := run(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	r, err := readRecord(recordPath(f.out, "a.de.raw"))
	if err != nil {
		t.Fatal(err)
	}
	r.Pins.WeightSHA256 = strings.Repeat("0", 64)
	if err := writeJSONAtomic(recordPath(f.out, "a.de.raw"), r); err != nil {
		t.Fatal(err)
	}
	cfg.Limit = 0
	if err := run(context.Background(), cfg); err == nil || !strings.Contains(err.Error(), "weight_sha256") {
		t.Fatalf("run = %v; want a refusal of the record", err)
	}
	if err := os.WriteFile(recordPath(f.out, "a.de.raw"), []byte("{cut"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run(context.Background(), cfg); err == nil || !strings.Contains(err.Error(), "a.de.raw") {
		t.Fatalf("run = %v; want a refusal that names the broken record", err)
	}
}

func TestRefusesAnotherWeight(t *testing.T) {
	f := newFixture(t)
	fake := newFake(filepath.Join(filepath.Dir(f.weight), "other.gguf"))
	if err := os.WriteFile(fake.modelPath, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := run(context.Background(), f.config(startFake(t, fake)))
	if err == nil || !strings.Contains(err.Error(), "serves") {
		t.Fatalf("run = %v", err)
	}
}

// The weight's realpath goes through a symlink, as a cold ~/models entry does.
func TestWeightSymlink(t *testing.T) {
	f := newFixture(t)
	link := filepath.Join(filepath.Dir(f.weight), "link.gguf")
	if err := os.Symlink(f.weight, link); err != nil {
		t.Fatal(err)
	}
	cfg := f.config(startFake(t, newFake(link)))
	cfg.Model, cfg.Limit = link, 1
	if err := run(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	r, err := readRecord(recordPath(f.out, "a.de.raw"))
	if err != nil {
		t.Fatal(err)
	}
	want, _ := filepath.EvalSymlinks(f.weight)
	if r.Pins.WeightPath != link || r.Pins.WeightRealpath != want {
		t.Fatalf("path %s, realpath %s", r.Pins.WeightPath, r.Pins.WeightRealpath)
	}
}

func TestStartsServer(t *testing.T) {
	f := newFixture(t)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	argsFile := filepath.Join(t.TempDir(), "args")
	t.Setenv("KOLIBRI_LOOPS_FAKE_ARGS", argsFile)
	cfg := f.config("")
	cfg.Server = exe
	cfg.ServerArgs = []string{"-ngl", "99", "--expert-cache", "128"}
	cfg.Limit = 2
	if err := run(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatal(err)
	}
	args := strings.Split(string(data), "\n")
	if args[0] != "-m" || args[1] != f.weight || args[2] != "--host" || args[3] != "127.0.0.1" || !strings.HasSuffix(string(data), "-ngl\n99\n--expert-cache\n128") {
		t.Fatalf("server args %q", args)
	}
	r, err := readRecord(recordPath(f.out, "a.de.raw"))
	if err != nil {
		t.Fatal(err)
	}
	if r.Pins.Attached || !slices.Equal(r.Pins.ServerArgs, cfg.ServerArgs) {
		t.Fatalf("pins %+v", r.Pins)
	}
	wantHash, _ := hashFile(exe)
	if r.Pins.ServerSHA256 != wantHash {
		t.Fatal("the record does not hash the started binary")
	}
	// pass-through flags cannot change what the tool pins
	cfg.ServerArgs = []string{"--port", "1"}
	if err := run(context.Background(), cfg); err == nil || !strings.Contains(err.Error(), "--port") {
		t.Fatalf("run = %v", err)
	}
}

func TestServerThatExitsWhileLoading(t *testing.T) {
	f := newFixture(t)
	cfg := f.config("")
	cfg.Server = "/usr/bin/false"
	if err := run(context.Background(), cfg); err == nil || !strings.Contains(err.Error(), "exited while loading") {
		t.Fatalf("run = %v", err)
	}
}

// llama-server lists no probability for a token that ends in an incomplete
// UTF-8 sequence, though `tokens` has it (seen on the tiny checkpoint).
func TestFragmentTokenKeepsItsPosition(t *testing.T) {
	f := newFixture(t)
	fake := newFake(f.weight)
	fake.fragments = map[int]bool{1003: true}
	if err := run(context.Background(), f.config(startFake(t, fake))); err != nil {
		t.Fatal(err)
	}
	r, err := readRecord(recordPath(f.out, "b.de.raw"))
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Probs) != len(r.GeneratedTokens) || r.Probs[3] != nil || r.Probs[2].ID != 1002 || r.Probs[4].ID != 1004 {
		t.Fatalf("probabilities are not aligned: %d entries, 3 is %v", len(r.Probs), r.Probs[3])
	}
	rs, err := readRecords(f.out)
	if err != nil {
		t.Fatal(err)
	}
	rep, err := buildReport(rs)
	if err != nil {
		t.Fatal(err)
	}
	if rep.TokensNoProb == 0 {
		t.Fatal("the report does not count tokens without a probability")
	}
}

func TestAlignProbs(t *testing.T) {
	entry := func(id int) tokenProb { return tokenProb{ID: id} }
	got, err := alignProbs([]int{1, 2, 3}, []tokenProb{entry(1), entry(3)})
	if err != nil || got[0].ID != 1 || got[1] != nil || got[2].ID != 3 {
		t.Fatalf("alignProbs = %v, %v", got, err)
	}
	if _, err := alignProbs([]int{1, 2}, []tokenProb{entry(2), entry(1)}); err == nil {
		t.Fatal("alignProbs accepted another order")
	}
	if _, err := alignProbs([]int{1}, []tokenProb{entry(1), entry(1)}); err == nil {
		t.Fatal("alignProbs accepted more entries than tokens")
	}
}

// Banning a fragment token leaves the measured distribution alone.
func TestLossBanKeepsTheDistribution(t *testing.T) {
	f := newFixture(t)
	fake := newFake(f.weight)
	fake.fragments = map[int]bool{0: true, 1: true}
	if err := run(context.Background(), f.config(startFake(t, fake))); err != nil {
		t.Fatal(err)
	}
	r, err := readRecord(recordPath(f.out, "x.en.text"))
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range r.Positions {
		if p.Bans != 2 || math.Abs(p.Logprob+math.Log(512)) > 1e-12 {
			t.Fatalf("position %+v", p)
		}
	}
	if !strings.Contains(string(r.Request), `"logit_bias":[[0,false],[1,false]]`) {
		t.Fatalf("the record's request is not the last one sent: %s", r.Request)
	}
	// every candidate's a fragment: the position stays unmeasured
	g := newFixture(t)
	all := newFake(g.weight)
	all.fragments = map[int]bool{}
	for i := range 512 {
		all.fragments[i] = true
	}
	all.gen = func(_ []int, n int) ([]int, string) { return distinct(2000, n), "limit" }
	err = run(context.Background(), g.config(startFake(t, all)))
	if err == nil || !strings.Contains(err.Error(), "no distribution") {
		t.Fatalf("run = %v", err)
	}
}

// The report needs nothing but the records.
func TestReportFromRecordsAlone(t *testing.T) {
	f := newFixture(t)
	fake := newFake(f.weight)
	fake.gen = func(p []int, n int) ([]int, string) {
		if p[0] == 452 {
			return seq([]int{1678, 13}, n, 0)[:n], "limit"
		}
		if p[0] == 325 {
			return []int{1000, 1001, 127906}, "eos"
		}
		return defaultGen(p, n)
	}
	if err := run(context.Background(), f.config(startFake(t, fake))); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{f.corpus, f.weight, f.server, runPath(f.out)} {
		if err := os.Remove(p); err != nil {
			t.Fatal(err)
		}
	}
	rs, err := readRecords(f.out)
	if err != nil {
		t.Fatal(err)
	}
	rep, err := buildReport(rs)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Records != len(fixtureItems()) || rep.Items != len(fixtureItems()) {
		t.Fatalf("records %d of %d", rep.Records, rep.Items)
	}
	var loops, n, eos int
	for _, c := range rep.Repetition {
		loops += c.Loops
		n += c.N
		eos += c.EOS
		if c.Form == formText {
			t.Fatal("a text in the repetition table")
		}
	}
	if loops != 1 || n != 10 || eos != 1 {
		t.Fatalf("loops %d, n %d, eos %d", loops, n, eos)
	}
	for _, p := range rep.PerRecord {
		if p.ID == "a.de.raw" && (!p.Looping || p.Start == nil || *p.Start != 0 || *p.Period != 2) {
			t.Fatalf("a.de.raw: %+v", p)
		}
		if p.ID != "a.de.raw" && (p.Looping || p.Start != nil || p.Period != nil) {
			t.Fatalf("%s: a generation without a loop has a start: %+v", p.ID, p)
		}
	}
	// answers: only pair a, whose accept is the fake's token 1001
	var correct, wrong, scored int
	for _, c := range rep.Answers {
		correct += c.Correct
		wrong += c.Wrong
		scored += c.Scored
	}
	if scored != 4 || correct != 3 || wrong != 1 { // the German raw prompt loops on other tokens
		t.Fatalf("scored %d, correct %d, wrong %d", scored, correct, wrong)
	}
	// loss: the fake's distribution is uniform over 512 tokens
	var ppls int
	for _, c := range rep.Loss {
		if c.PPL == nil || math.Abs(*c.PPL-512) > 1e-6 {
			t.Fatalf("loss %+v", c)
		}
		ppls += c.Positions
	}
	if ppls != 8+6 {
		t.Fatalf("%d loss positions", ppls)
	}
	// the same records give the same bytes
	var a, b bytes.Buffer
	writeReport(&a, rep)
	rs2, _ := readRecords(f.out)
	rep2, err := buildReport(rs2)
	if err != nil {
		t.Fatal(err)
	}
	writeReport(&b, rep2)
	if !bytes.Equal(a.Bytes(), b.Bytes()) {
		t.Fatal("two reports of the same records differ")
	}
	for _, want := range []string{"Reference logits", "BF16", "Tool use"} {
		if !strings.Contains(a.String(), want) {
			t.Fatalf("the report does not say that %s is not done", want)
		}
	}
}

func TestReportRefuses(t *testing.T) {
	f := newFixture(t)
	if err := run(context.Background(), f.config(startFake(t, newFake(f.weight)))); err != nil {
		t.Fatal(err)
	}
	load := func(t *testing.T) []Record {
		rs, err := readRecords(f.out)
		if err != nil {
			t.Fatal(err)
		}
		return rs
	}
	index := func(rs []Record, id string) int {
		return slices.IndexFunc(rs, func(r Record) bool { return r.ID == id })
	}
	for _, c := range []struct {
		name, want string
		change     func(rs []Record)
	}{
		{"records of two runs", "different runs", func(rs []Record) { rs[3].Pins.WeightSHA256 = "x" }},
		{"a generated token edited", "not greedy", func(rs []Record) {
			r := &rs[index(rs, "a.de.raw")]
			r.Probs[4].ID = 99
			r.GeneratedTokens[4] = 99
		}},
		{"a token count edited", "probabilities for", func(rs []Record) {
			r := &rs[index(rs, "a.de.raw")]
			r.GeneratedTokens = r.GeneratedTokens[:10]
		}},
		{"a prompt edited", "prompt_tokens", func(rs []Record) {
			rs[index(rs, "a.de.raw")].PromptTokens[0]++
		}},
		{"a loss position dropped", "positions", func(rs []Record) {
			r := &rs[index(rs, "x.de.text")]
			r.Positions = r.Positions[1:]
		}},
		{"a loss above certainty", "log probability", func(rs []Record) {
			rs[index(rs, "x.de.text")].Positions[2].Logprob = 0.5
		}},
		{"a NaN loss", "log probability", func(rs []Record) {
			rs[index(rs, "x.de.text")].Positions[2].Logprob = math.NaN()
		}},
		{"a generated probability above certainty", "log probability", func(rs []Record) {
			rs[index(rs, "a.de.raw")].Probs[2].Logprob = 0.5
		}},
		{"a partial distribution", "the vocabulary has", func(rs []Record) {
			rs[index(rs, "x.de.text")].Positions[2].Candidates = 100
		}},
		{"a text's settings", "other settings", func(rs []Record) {
			rs[index(rs, "x.de.text")].ResolvedSettings = []byte(`{"seed":43}`)
		}},
		{"a text's prompt", "last position", func(rs []Record) {
			rs[index(rs, "x.de.text")].PromptTokens[0]++
		}},
		{"a truncated prompt", "truncated", func(rs []Record) {
			rs[index(rs, "a.de.raw")].Truncated = true
		}},
		{"a truncated text", "truncated", func(rs []Record) {
			rs[index(rs, "x.de.text")].Truncated = true
		}},
		{"a token count that differs from the server's", "tokens_predicted", func(rs []Record) {
			rs[index(rs, "a.de.raw")].TokensPredicted = 31
		}},
		{"another n_predict than pinned", "pinned settings", func(rs []Record) {
			r := &rs[index(rs, "b.de.raw")]
			r.Request = bytes.Replace(r.Request, []byte(`"n_predict":32`), []byte(`"n_predict":16`), 1)
		}},
		{"another seed than pinned", "pinned settings", func(rs []Record) {
			r := &rs[index(rs, "b.de.raw")]
			r.Request = bytes.Replace(r.Request, []byte(`"seed":42`), []byte(`"seed":43`), 1)
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			rs := load(t)
			c.change(rs)
			if _, err := buildReport(rs); err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("buildReport = %v; want %q", err, c.want)
			}
		})
	}
	if _, err := buildReport(nil); err == nil {
		t.Fatal("a report without records")
	}
}

// Two files cannot count one prompt twice.
func TestReportRefusesRenamedRecord(t *testing.T) {
	f := newFixture(t)
	if err := run(context.Background(), f.config(startFake(t, newFake(f.weight)))); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(recordPath(f.out, "a.de.raw"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(recordPath(f.out, "copy"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := readRecords(f.out); err == nil || !strings.Contains(err.Error(), "holds record a.de.raw") {
		t.Fatalf("readRecords = %v", err)
	}
}

// The numbers and the printed rule must be one rule: a report on records of
// another rule is refused, not computed under that rule's label.
func TestReportRefusesAnotherRule(t *testing.T) {
	f := newFixture(t)
	cfg := f.config(startFake(t, newFake(f.weight)))
	cfg.Settings.Rule = Rule{MaxPeriod: 1, MinRepeats: 50, MinRegion: 200}
	if err := run(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	rs, err := readRecords(f.out)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := buildReport(rs); err == nil || !strings.Contains(err.Error(), "loop rule") {
		t.Fatalf("buildReport = %v; want a refusal that names the rule", err)
	}
}

// A cell or a control without a loop has no start. Start 0 means the first token,
// so 0 here would read as a loop from the start.
func TestNoLoopHasNoStart(t *testing.T) {
	f := newFixture(t)
	items := fixtureItems()
	for i := range items {
		if items[i].ID == "b.de.raw" {
			items[i].Control, items[i].Source = controlNonLoop, "docs/real-checkpoint.md"
		}
	}
	writeCorpus(t, f.corpus, items)
	if err := run(context.Background(), f.config(startFake(t, newFake(f.weight)))); err != nil {
		t.Fatal(err)
	}
	rs, err := readRecords(f.out)
	if err != nil {
		t.Fatal(err)
	}
	rep, err := buildReport(rs)
	if err != nil {
		t.Fatal(err)
	}
	var row *RepetitionCell
	for i, c := range rep.Repetition {
		if c.Loops == 0 {
			row = &rep.Repetition[i]
		}
	}
	if row == nil || row.MedianStart != nil || row.MaxStart != nil {
		t.Fatalf("a cell without loops: %+v", row)
	}
	if c := rep.Controls[0]; c.Looping || c.Start != nil || c.Period != nil {
		t.Fatalf("a control without a loop: %+v", c)
	}
	data, err := json.Marshal(rep)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"median_start":null`) || !strings.Contains(string(data), `"max_start":null`) {
		t.Fatalf("report.json: %s", data)
	}
	var out bytes.Buffer
	writeReport(&out, rep)
	if !strings.Contains(out.String(), "| 0.000 | - | - | [] |") {
		t.Fatalf("the table prints a start without a loop:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "| b.de.raw | "+rep.Controls[0].Split+" | non-loop | false | - | - |") {
		t.Fatalf("the control table prints a start without a loop:\n%s", out.String())
	}
}

// A generation the limit stops before a loop can qualify is censored, not cured.
func TestReportCountsCensored(t *testing.T) {
	f := newFixture(t)
	fake := newFake(f.weight)
	const n, p = 32, 2 // the region needs 16 tokens, so a loop can start at 16 at the latest
	fake.gen = func(prompt []int, nPredict int) ([]int, string) {
		switch prompt[0] {
		case 452: // detectable: starts on the last token that allows it
			return concat(distinct(2000, n-16), seq([]int{5, 6}, 8, 0)), "limit"
		case 325: // the same loop, one token later: not detectable
			return concat(distinct(2000, n-15), seq([]int{5, 6}, 7, 1)), "limit"
		}
		return []int{1000, 1001, 127906}, "eos"
	}
	if err := run(context.Background(), f.config(startFake(t, fake))); err != nil {
		t.Fatal(err)
	}
	rs, err := readRecords(f.out)
	if err != nil {
		t.Fatal(err)
	}
	rep, err := buildReport(rs)
	if err != nil {
		t.Fatal(err)
	}
	var loops, censored, eos int
	for _, c := range rep.Repetition {
		loops += c.Loops
		censored += c.Censored
		eos += c.EOS
	}
	if loops != 1 || censored != 1 || eos != 8 {
		t.Fatalf("loops %d, censored %d, eos %d", loops, censored, eos)
	}
	var out bytes.Buffer
	writeReport(&out, rep)
	for _, want := range []string{"Limit, no loop (censored)", "censored by the limit", "lower bound"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("report misses %q:\n%s", want, out.String())
		}
	}
}

// A target with probability 0 has no finite loss: the report says so instead of
// leaving it out.
func TestReportZeroProbability(t *testing.T) {
	f := newFixture(t)
	if err := run(context.Background(), f.config(startFake(t, newFake(f.weight)))); err != nil {
		t.Fatal(err)
	}
	rs, err := readRecords(f.out)
	if err != nil {
		t.Fatal(err)
	}
	i := slices.IndexFunc(rs, func(r Record) bool { return r.ID == "x.de.text" })
	rs[i].Positions[0].Logprob = -math.MaxFloat32
	rep, err := buildReport(rs)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range rep.Loss {
		if c.Lang == "de" && (c.Zero != 1 || c.PPL != nil || c.MeanNLL != nil) {
			t.Fatalf("loss %+v", c)
		}
	}
}

// Negative controls: each is a mutation that the checks must catch
// (docs/loop-baseline.md, "Negative controls").

// M72: a detector that only knows period 1 misses the documented period-2 and
// longer loops.
func TestControlM72PeriodOneDetector(t *testing.T) {
	mutated := Rule{MaxPeriod: 1, MinRepeats: FixedRule.MinRepeats, MinRegion: FixedRule.MinRegion}
	dass := seq([]int{5, 6}, 128, 0)
	if !FixedRule.Detect(dass, true).Looping {
		t.Fatal("the fixed rule misses period 2")
	}
	if mutated.Detect(dass, true).Looping {
		t.Fatal("M72 did not change the result: the fixture cannot tell the rules apart")
	}
}

// M74a: a server that does not decode greedily never leaves a record.
func TestControlM74aNotGreedy(t *testing.T) {
	f := newFixture(t)
	fake := newFake(f.weight)
	fake.notGreedy = true
	err := run(context.Background(), f.config(startFake(t, fake)))
	if err == nil || !strings.Contains(err.Error(), "not greedy") {
		t.Fatalf("run = %v", err)
	}
	if got := recordNames(t, f.out); len(got) != 0 {
		t.Fatalf("records %v", got)
	}
}

// M74b: a server that lists only part of the distribution gives no loss, rather
// than a bound passed off as one.
func TestControlM74bPartialDistribution(t *testing.T) {
	f := newFixture(t)
	fake := newFake(f.weight)
	fake.truncateVocab = 100
	err := run(context.Background(), f.config(startFake(t, fake)))
	if err == nil || !strings.Contains(err.Error(), "the vocabulary has") {
		t.Fatalf("run = %v", err)
	}
	if _, statErr := os.Stat(recordPath(f.out, "x.de.text")); !os.IsNotExist(statErr) {
		t.Fatalf("a text record exists: %v", statErr)
	}
}

// M73: a looser rule counts the near-misses as loops.
func TestControlM73LooseRule(t *testing.T) {
	loose := Rule{MaxPeriod: 64, MinRepeats: 2, MinRegion: 4}
	shortRun := concat(distinct(1, 20), seq([]int{7}, 15, 0))
	twoUnits := seq(unit(40), 2, 0)
	for _, tokens := range [][]int{shortRun, twoUnits} {
		if FixedRule.Detect(tokens, true).Looping {
			t.Fatal("the fixed rule counts a near-miss")
		}
		if !loose.Detect(tokens, true).Looping {
			t.Fatal("M73 did not change the result: the fixture cannot tell the rules apart")
		}
	}
}

// M74f: llama-server writes a NaN as null, and Go would read null as 0, which is
// probability 1. A NaN decode must stop a run, not score perfectly or look like a
// confident loop.
func TestControlM74fNaNLogits(t *testing.T) {
	f := newFixture(t)
	fake := newFake(f.weight)
	fake.nanLogits = true
	err := run(context.Background(), f.config(startFake(t, fake)))
	if err == nil || !strings.Contains(err.Error(), "log probability is null") {
		t.Fatalf("run = %v", err)
	}
	if got := recordNames(t, f.out); len(got) != 0 {
		t.Fatalf("records %v", got)
	}
	// and the distribution of a text, on its own
	g := newFixture(t)
	writeCorpus(t, g.corpus, slices.DeleteFunc(fixtureItems(), func(it Item) bool { return it.Form != formText }))
	fakeG := newFake(g.weight)
	fakeG.nanLogits = true
	err = run(context.Background(), g.config(startFake(t, fakeG)))
	if err == nil || !strings.Contains(err.Error(), "log probability is null") {
		t.Fatalf("run on texts = %v", err)
	}
	if got := recordNames(t, g.out); len(got) != 0 {
		t.Fatalf("text records %v", got)
	}
}

// A record edited to hold a null is refused when read, so a report cannot read it
// as probability 1.
func TestRecordWithNullLogprob(t *testing.T) {
	f := newFixture(t)
	if err := run(context.Background(), f.config(startFake(t, newFake(f.weight)))); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ id, old string }{
		{"a.de.raw", `"logprob":-0.1,"top_logprobs"`},
		{"a.de.raw", `{"id":1679,"logprob":-3}`},
		{"x.de.text", `"logprob":-6.238324625039508,"top_id"`},
	} {
		path := recordPath(f.out, c.id)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(data), c.old) {
			t.Fatalf("%s holds no %s", c.id, c.old)
		}
		edited := strings.Replace(string(data), c.old, strings.NewReplacer("-0.1", "null", "-3", "null", "-6.238324625039508", "null").Replace(c.old), 1)
		if err := os.WriteFile(path, []byte(edited), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := readRecords(f.out); err == nil || !strings.Contains(err.Error(), "log probability is null") {
			t.Fatalf("readRecords with %s nulled = %v", c.old, err)
		}
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// What the decoding must let through: probability 1, its float32 rounding, and a
// position without an entry, which is a fragment token and no log probability.
func TestLogprobAtCertaintyIsAccepted(t *testing.T) {
	var e topEntry
	if err := json.Unmarshal([]byte(`{"id":1,"logprob":1.1920928955078125e-07}`), &e); err != nil {
		t.Fatalf("float32 rounding of probability 1 was refused: %v", err)
	}
	if err := json.Unmarshal([]byte(`{"id":1,"logprob":0}`), &e); err != nil {
		t.Fatalf("probability 1 was refused: %v", err)
	}
	if err := json.Unmarshal([]byte(`{"id":1}`), &e); err == nil {
		t.Fatal("a missing log probability was accepted")
	}
	var prob struct {
		Probs []*tokenProb `json:"probs"`
	}
	if err := json.Unmarshal([]byte(`{"probs":[null]}`), &prob); err != nil || prob.Probs[0] != nil {
		t.Fatalf("a position without an entry: %v, %v", prob.Probs, err)
	}
}

// M74c, d and e: a drifting tokenizer, a server that resolves another seed and a
// template that ignores the thinking mode are refused before they become records.
func TestControlM74cdeServerDrift(t *testing.T) {
	for _, c := range []struct {
		name, want string
		mutate     func(*fakeServer)
		corpus     func(items []Item) []Item
	}{
		{"M74c tokenizer", "the docs record", func(f *fakeServer) { f.tokenizeOffset = 1 }, func(items []Item) []Item {
			items[0].ExpectTokens = documentedTokens[items[0].Prompt]
			return items
		}},
		{"M74d seed", "other settings", func(f *fakeServer) { f.ignoreSeed = true }, nil},
		{"M74e template", "think block", func(f *fakeServer) { f.wrongThinkBlock = true }, nil},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := newFixture(t)
			if c.corpus != nil {
				writeCorpus(t, f.corpus, c.corpus(fixtureItems()))
			}
			fake := newFake(f.weight)
			c.mutate(fake)
			err := run(context.Background(), f.config(startFake(t, fake)))
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("run = %v; want %q", err, c.want)
			}
		})
	}
}

func TestCorpus(t *testing.T) {
	c, err := loadCorpus(realCorpus)
	if err != nil {
		t.Fatal(err)
	}
	if c.SHA256 != corpusSHA256 {
		t.Fatalf("corpus sha256 %s; the corpus and its split are pinned", c.SHA256)
	}
	cells := map[string]int{}
	loopControls := map[string]map[string]int{splitTuning: {}, splitHeldOut: {}}
	for _, it := range c.Items {
		cells[it.Split+"/"+it.Form+"/"+it.Lang]++
		if it.Control == controlLoop {
			loopControls[it.Split][it.Lang]++
		}
		if it.Source != "" {
			checkDocumented(t, it)
		}
	}
	// balanced: each cell has its counterpart in the other language
	for key, n := range cells {
		other := strings.Replace(key, "/de", "/en", 1)
		if other == key {
			other = strings.Replace(key, "/en", "/de", 1)
		}
		if cells[other] != n {
			t.Errorf("%s has %d items, %s has %d", key, n, other, cells[other])
		}
	}
	for _, split := range []string{splitTuning, splitHeldOut} {
		if loopControls[split]["de"] == 0 {
			t.Errorf("no German looping control in %s", split)
		}
		for _, form := range []string{formRaw, formChat, formChatThink, formText} {
			if cells[split+"/"+form+"/de"] == 0 {
				t.Errorf("nothing in %s/%s", split, form)
			}
		}
	}
	if loopControls[splitTuning]["en"]+loopControls[splitHeldOut]["en"] == 0 {
		t.Error("no English looping control")
	}
}

// checkDocumented makes sure a control's prompt is the documented one.
func checkDocumented(t *testing.T, it Item) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("../..", it.Source))
	if err != nil {
		t.Fatal(err)
	}
	if strings.HasSuffix(it.Source, ".json") {
		var v any
		if err := json.Unmarshal(data, &v); err != nil {
			t.Fatal(err)
		}
		if !jsonHolds(v, it.Prompt) {
			t.Errorf("%s: the prompt is not in %s", it.ID, it.Source)
		}
		return
	}
	escaped := strings.NewReplacer("\n", `\n`, "|", `\|`).Replace(it.Prompt)
	if !strings.Contains(string(data), `"`+escaped+`"`) {
		t.Errorf("%s: %q is not in %s", it.ID, escaped, it.Source)
	}
}

func jsonHolds(v any, want string) bool {
	switch x := v.(type) {
	case string:
		return x == want
	case []any:
		return slices.ContainsFunc(x, func(e any) bool { return jsonHolds(e, want) })
	case map[string]any:
		for _, e := range x {
			if jsonHolds(e, want) {
				return true
			}
		}
	}
	return false
}

// controlEvidence is what makes each control's label true: the words its source
// has for the run. A label that moves has to move here too, against this text.
var controlEvidence = map[string]struct{ label, snippet string }{
	"capital-germany.de.raw":  {controlLoop, `" Deutschland Deutschland Deutschland …"`},
	"capital-germany.en.raw":  {controlNonLoop, `" Berlin. Berlin is the largest city in Germany and the capital."`},
	"rhine-clause.de.raw":     {controlLoop, `", der, der, der …"`},
	"capitals-context.de.raw": {controlLoop, `" Berlin. Die Hauptstadt von Frankreich ist Paris. Die Hauptstadt von Italien ist Rom. …"`},
	"capitals-context.en.raw": {controlNonLoop, `" Berlin. The capital of Spain is Madrid.`},
	"berlin-lead.de.raw":      {controlLoop, `" die größte Stadt Deutschlands und ein Land der Bundesrepublik Deutschland. Mit rund 3,7 Millionen`},
	"eot-capital.de.raw":      {controlLoop, `" die Hauptstadt von Deutschland ist die Hauptstadt von Deutschland …"`},
	"eot-capital.en.raw":      {controlLoop, `" a capital of a capital of a capital of …"`},
	"question-capital.de.raw": {controlNonLoop, `" Berlin\n\nDie Hauptstadt von Deutschland ist Berlin.`},
	"rhine-context.en.raw":    {controlNonLoop, `", it passes through several countries`},
	"think-sky.de.chat-think": {controlNonLoop, `528 tokens`},
	// no row: the source says this in prose, on a 16-token run
	"moe-explain.de.chat": {controlNonLoop, "(`de-chat`, and the 528-token German answer"},
}

// A control's label has to agree with what its source records, on the line that
// holds the prompt. The hash pin alone would let a flipped label through.
func TestControlLabelsMatchTheirSource(t *testing.T) {
	c, err := loadCorpus(realCorpus)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("../../docs/real-checkpoint.md")
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(string(data), "\n")
	seen := 0
	for _, it := range c.Items {
		if it.Control == "" {
			continue
		}
		seen++
		ev, ok := controlEvidence[it.ID]
		if !ok || ev.label != it.Control {
			t.Errorf("%s is labelled %s; its evidence says %+v", it.ID, it.Control, ev)
			continue
		}
		prompt := `"` + strings.NewReplacer("\n", `\n`, "|", `\|`).Replace(it.Prompt) + `"`
		if it.ID == "moe-explain.de.chat" {
			prompt = ""
		}
		if !slices.ContainsFunc(lines, func(l string) bool { return strings.Contains(l, prompt) && strings.Contains(l, ev.snippet) }) {
			t.Errorf("%s: no line of the source holds %s and %s", it.ID, prompt, ev.snippet)
		}
	}
	if seen != len(controlEvidence) {
		t.Errorf("%d controls, evidence for %d", seen, len(controlEvidence))
	}
}

// Nothing a tuned intervention has seen may come back in the held-out split: the
// shapes the reviewer found in the first split, checked on the committed corpus.
func TestSplitKeepsFactsApart(t *testing.T) {
	c, err := loadCorpus(realCorpus)
	if err != nil {
		t.Fatal(err)
	}
	byID := map[string]Item{}
	byPrompt := map[string]string{}
	bySplit := map[string][]Item{}
	for _, it := range c.Items {
		byID[it.ID] = it
		bySplit[it.Split] = append(bySplit[it.Split], it)
		bare := strings.ReplaceAll(it.Prompt, "<|endoftext|>", "")
		if s, ok := byPrompt[bare]; ok && s != it.Split {
			t.Errorf("the prompt of %s is on both sides", it.ID)
		}
		byPrompt[bare] = it.Split
	}
	for _, ids := range [][2]string{
		{"think-planet.de.chat-think", "planet-largest.de.chat"},
		{"eot-capital.de.raw", "capital-germany.de.raw"},
		{"capital-france.de.raw", "capitals-context.de.raw"},
		{"berlin-lead.de.raw", "capital-germany.de.raw"},
		{"think-capital.de.chat-think", "question-capital.de.raw"},
		{"rhine-clause.de.raw", "rhine-context.de.raw"},
	} {
		if byID[ids[0]].Split != byID[ids[1]].Split {
			t.Errorf("%s and %s share a fact across the split", ids[0], ids[1])
		}
	}
	for _, tuned := range bySplit[splitTuning] {
		for _, held := range bySplit[splitHeldOut] {
			a, b := strings.ToLower(tuned.Prompt), strings.ToLower(held.Prompt)
			if strings.Contains(a, b) || strings.Contains(b, a) {
				t.Errorf("%s and %s overlap across the split", tuned.ID, held.ID)
			}
			if tuned.Answer != nil && held.Answer != nil && slices.ContainsFunc(tuned.Answer.Accept, func(t string) bool { return slices.Contains(held.Answer.Accept, t) }) {
				t.Errorf("%s and %s share an answer across the split", tuned.ID, held.ID)
			}
		}
	}
	// a looping control on each side, by construction: two groups hold them
	sides := map[string]bool{}
	for _, it := range c.Items {
		if it.Control == controlLoop {
			sides[it.Split] = true
		}
	}
	if len(sides) != 2 {
		t.Errorf("looping controls on %v only", sides)
	}
}

func TestValidateCorpus(t *testing.T) {
	for _, c := range []struct {
		name, want string
		change     func(items []Item) []Item
	}{
		{"a split moved by hand", "the rule says", func(items []Item) []Item {
			if items[0].Split == splitTuning {
				items[0].Split = splitHeldOut
			} else {
				items[0].Split = splitTuning
			}
			return items
		}},
		{"a missing language", "needs", func(items []Item) []Item { return items[1:] }},
		{"a duplicate", "duplicate", func(items []Item) []Item { return append(items, items[0]) }},
		{"an unknown form", "id should be", func(items []Item) []Item { items[0].Form = "other"; return items }},
		{"a pair in two strata", "strata", func(items []Item) []Item { items[0].Stratum = stratumOpen; return items }},
		{"no group", "no group", func(items []Item) []Item { items[0].Group = ""; return items }},
		{"a pair in two groups", "groups", func(items []Item) []Item { items[0].Group = "other"; return items }},
		{"the same prompt in two groups", "holds the prompt", func(items []Item) []Item {
			items[0].Prompt = "Warum?" // pair b's German chat prompt
			return items
		}},
		{"a prompt inside another group's", "holds the prompt", func(items []Item) []Item {
			items[0].Prompt = "Es war einmal" // holds pair b's "Es war"
			return items
		}},
		{"a prompt behind a special token", "holds the prompt", func(items []Item) []Item {
			items[0].Prompt = "<|endoftext|>Warum?"
			return items
		}},
		{"the same answer in two groups", "the same answer", func(items []Item) []Item {
			for i := range items {
				if items[i].Pair == "b" {
					items[i].Answer = &Answer{Accept: []string{"1001"}}
				}
			}
			return items
		}},
		{"an answer in another group's prompt", "holds the answer", func(items []Item) []Item {
			for i := range items {
				if items[i].ID == "b.de.raw" {
					items[i].Prompt = "Vielleicht 1001"
				}
			}
			return items
		}},
		{"a control without a source", "source", func(items []Item) []Item { items[0].Control = controlLoop; return items }},
		{"an answer on a text", "a text has no answer", func(items []Item) []Item {
			for i := range items {
				if items[i].Form == formText {
					items[i].Answer = &Answer{Accept: []string{"x"}}
				}
			}
			return items
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			if err := validateCorpus(c.change(fixtureItems())); err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("validateCorpus = %v; want %q", err, c.want)
			}
		})
	}
}

// The whole corpus runs through the fake, and the report recomputes it.
func TestWholeCorpusAgainstFake(t *testing.T) {
	f := newFixture(t)
	data, err := os.ReadFile(realCorpus)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(f.corpus, data, 0o644); err != nil {
		t.Fatal(err)
	}
	fake := newFake(f.weight)
	if err := run(context.Background(), f.config(startFake(t, fake))); err != nil {
		t.Fatal(err)
	}
	rs, err := readRecords(f.out)
	if err != nil {
		t.Fatal(err)
	}
	rep, err := buildReport(rs)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Records != 112 || rep.Items != 112 {
		t.Fatalf("records %d of %d", rep.Records, rep.Items)
	}
	// the fake loops exactly on the prompt with the documented German IDs
	loops := 0
	for _, c := range rep.Repetition {
		loops += c.Loops
	}
	if loops != 1 {
		t.Fatalf("%d loops; want 1", loops)
	}
	var out bytes.Buffer
	writeReport(&out, rep)
	for _, want := range []string{"| tuning | raw | de |", "| held-out | chat-think | en |", "capital-germany.de.raw"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("report misses %q:\n%s", want, out.String())
		}
	}
	if strings.Contains(out.String(), fmt.Sprintf("| %s |", formText)) {
		t.Error("a text in a form table")
	}
}

// Pass-through flags cannot take over what the tool sets, in either spelling.
func TestServerFlagGuard(t *testing.T) {
	for _, arg := range []string{"-m", "--model", "--host", "--port", "-m=x", "--model=x", "--host=0.0.0.0", "--port=1"} {
		cfg := runConfig{Server: "unused", Model: "unused", ServerArgs: []string{arg}}
		if err := run(context.Background(), cfg); err == nil || !strings.Contains(err.Error(), "is set by the tool") {
			t.Errorf("%s: %v", arg, err)
		}
	}
	for _, arg := range []string{"-ngl", "--expert-cache=128", "--host-name=x"} {
		if slices.Contains(forbiddenServerArgs, serverFlagName(arg)) {
			t.Errorf("%s is refused", arg)
		}
	}
}

// A server that accepts the health request and never answers does not hold the
// load past its deadline.
func TestHealthyEndsAtTheDeadline(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { <-release }))
	defer srv.Close()
	defer close(release)
	start := time.Now()
	if healthy(context.Background(), srv.URL, start.Add(100*time.Millisecond)) {
		t.Fatal("a server that never answers is healthy")
	}
	if d := time.Since(start); d > 2*time.Second {
		t.Fatalf("took %v", d)
	}
}

func TestCheckResolvedHoldsPostSamplingProbs(t *testing.T) {
	req := greedyRequest([]int{1}, 1, 42, 2)
	for _, c := range []struct {
		field string
		want  bool
	}{
		{`"post_sampling_probs":false,`, true},
		{`"post_sampling_probs":true,`, false},
		{``, false},
	} {
		settings := fmt.Sprintf(`{%s"seed":%d,"temperature":%v,"repeat_penalty":%v,"n_probs":%d,"samplers":["temperature"]}`,
			c.field, req.Seed, req.Temperature, req.RepeatPenalty, req.NProbs)
		resp := completionResponse{GenerationSettings: json.RawMessage(settings), TokensEvaluated: 1}
		if err := checkResolved(req, resp); (err == nil) != c.want {
			t.Errorf("%q: %v", c.field, err)
		}
	}
}

func TestLoadCorpusRefusesASecondValueOnALine(t *testing.T) {
	data, err := os.ReadFile(realCorpus)
	if err != nil {
		t.Fatal(err)
	}
	line, _, _ := bytes.Cut(data, []byte("\n"))
	for _, tail := range []string{` {"unseen":1}`, ` garbage`, ` {}`} {
		path := filepath.Join(t.TempDir(), "corpus.jsonl")
		if err := os.WriteFile(path, append(append(slices.Clone(line), tail...), '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := loadCorpus(path); err == nil || !strings.Contains(err.Error(), "more than one JSON value") {
			t.Errorf("%q: %v", tail, err)
		}
	}
}
