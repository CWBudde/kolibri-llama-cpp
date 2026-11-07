package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"
)

// client talks to one llama-server. It sends one request at a time.
type client struct {
	base    string
	http    *http.Client
	timeout time.Duration
}

func newClient(base string, timeout time.Duration) *client {
	return &client{base: strings.TrimRight(base, "/"), http: &http.Client{}, timeout: timeout}
}

func (c *client) do(ctx context.Context, method, path string, body []byte, out any) error {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, method, c.base+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("%s %s: %s: %s", method, path, resp.Status, msg)
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("%s %s: %v", method, path, err)
	}
	return nil
}

// serverProps is what the pins take from the server.
type serverProps struct {
	ModelPath    string `json:"model_path"`
	BuildInfo    string `json:"build_info"`
	ChatTemplate string `json:"chat_template"`
}

func (c *client) props(ctx context.Context) (serverProps, error) {
	var p serverProps
	err := c.do(ctx, http.MethodGet, "/props", nil, &p)
	return p, err
}

// nVocab is the model's vocabulary size, from /v1/models.
func (c *client) nVocab(ctx context.Context) (int, error) {
	var m struct {
		Data []struct {
			Meta struct {
				NVocab int `json:"n_vocab"`
			} `json:"meta"`
		} `json:"data"`
	}
	if err := c.do(ctx, http.MethodGet, "/v1/models", nil, &m); err != nil {
		return 0, err
	}
	if len(m.Data) == 0 || m.Data[0].Meta.NVocab <= 0 {
		return 0, fmt.Errorf("/v1/models: no n_vocab")
	}
	return m.Data[0].Meta.NVocab, nil
}

// tokenizeRequest takes the prompt as the docs' runs did: no BOS is added, and
// special tokens such as <|endoftext|> are parsed.
type tokenizeRequest struct {
	Content      string `json:"content"`
	AddSpecial   bool   `json:"add_special"`
	ParseSpecial bool   `json:"parse_special"`
}

func (c *client) tokenize(ctx context.Context, req []byte) ([]int, error) {
	var r struct {
		Tokens []int `json:"tokens"`
	}
	if err := c.do(ctx, http.MethodPost, "/tokenize", req, &r); err != nil {
		return nil, err
	}
	if len(r.Tokens) == 0 {
		return nil, fmt.Errorf("/tokenize: no tokens")
	}
	return r.Tokens, nil
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// templateRequest renders a user message. ReasoningEffort "none" turns thinking
// off; left out, the template's default (thinking on) applies.
type templateRequest struct {
	Messages        []chatMessage `json:"messages"`
	ReasoningEffort string        `json:"reasoning_effort,omitempty"`
}

func (c *client) applyTemplate(ctx context.Context, req []byte) (string, error) {
	var r struct {
		Prompt string `json:"prompt"`
	}
	if err := c.do(ctx, http.MethodPost, "/apply-template", req, &r); err != nil {
		return "", err
	}
	return r.Prompt, nil
}

// completionRequest fixes every sampler setting, so nothing depends on the
// server's defaults. samplers ["temperature"] at temperature 0 is greedy.
// cache_prompt is off: batch sizes can change logits, so each prompt is decoded
// the same way whatever ran before it. post_sampling_probs is off: the log
// probabilities are the softmax of the logits, not of the sampler's output.
type completionRequest struct {
	Prompt            []int    `json:"prompt"`
	NPredict          int      `json:"n_predict"`
	Seed              int      `json:"seed"`
	Temperature       float64  `json:"temperature"`
	DynatempRange     float64  `json:"dynatemp_range"`
	TopK              int      `json:"top_k"`
	TopP              float64  `json:"top_p"`
	MinP              float64  `json:"min_p"`
	TypicalP          float64  `json:"typical_p"`
	XTCProbability    float64  `json:"xtc_probability"`
	RepeatLastN       int      `json:"repeat_last_n"`
	RepeatPenalty     float64  `json:"repeat_penalty"`
	PresencePenalty   float64  `json:"presence_penalty"`
	FrequencyPenalty  float64  `json:"frequency_penalty"`
	DryMultiplier     float64  `json:"dry_multiplier"`
	Mirostat          int      `json:"mirostat"`
	Samplers          []string `json:"samplers"`
	NKeep             int      `json:"n_keep"`
	NProbs            int      `json:"n_probs"`
	PostSamplingProbs bool     `json:"post_sampling_probs"`
	Stream            bool     `json:"stream"`
	CachePrompt       bool     `json:"cache_prompt"`
	ReturnTokens      bool     `json:"return_tokens"`
	IgnoreEOS         bool     `json:"ignore_eos"`
	Stop              []string `json:"stop"`
	// LogitBias bans tokens from sampling, as [id, false] pairs. The log
	// probabilities stay those of the raw logits. Only measureLoss uses it.
	LogitBias [][2]any `json:"logit_bias,omitempty"`
}

func greedyRequest(prompt []int, nPredict, seed, nProbs int) completionRequest {
	return completionRequest{
		Prompt: prompt, NPredict: nPredict, Seed: seed,
		TopP: 1, TypicalP: 1, RepeatPenalty: 1,
		Samplers: []string{"temperature"}, NProbs: nProbs,
		ReturnTokens: true, Stop: []string{},
	}
}

// topEntry is one candidate of a token's distribution.
type topEntry struct {
	ID      int     `json:"id"`
	Logprob float64 `json:"logprob"`
}

// maxLogprob allows for float32 rounding of a probability at 1, which a greedy
// loop reaches. Anything above it is not a log probability.
const maxLogprob = 1e-4

// checkLogprob refuses what cannot be a log probability.
func checkLogprob(v float64) error {
	if math.IsNaN(v) || v > maxLogprob {
		return fmt.Errorf("log probability %g", v)
	}
	return nil
}

// decodeLogprob refuses a null or missing value. llama-server's JSON library
// writes NaN as null, and Go would decode null into a float as 0, which reads as
// probability 1. A NaN decode must stop a run, not score perfectly.
func decodeLogprob(v *float64) (float64, error) {
	if v == nil {
		return 0, fmt.Errorf("log probability is null or missing, as the server writes NaN")
	}
	return *v, checkLogprob(*v)
}

func (e *topEntry) UnmarshalJSON(data []byte) error {
	var a struct {
		ID      int      `json:"id"`
		Logprob *float64 `json:"logprob"`
	}
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	lp, err := decodeLogprob(a.Logprob)
	if err != nil {
		return fmt.Errorf("candidate %d: %w", a.ID, err)
	}
	e.ID, e.Logprob = a.ID, lp
	return nil
}

// tokenProb is one generated token with its log probability and the candidates
// the server returned, most likely first. llama-server lists no entry for a
// token that ends in an incomplete UTF-8 sequence, in `tokens` it does.
type tokenProb struct {
	ID      int        `json:"id"`
	Logprob float64    `json:"logprob"`
	Top     []topEntry `json:"top_logprobs"`
}

func (p *tokenProb) UnmarshalJSON(data []byte) error {
	var a struct {
		ID      int        `json:"id"`
		Logprob *float64   `json:"logprob"`
		Top     []topEntry `json:"top_logprobs"`
	}
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	lp, err := decodeLogprob(a.Logprob)
	if err != nil {
		return fmt.Errorf("token %d: %w", a.ID, err)
	}
	p.ID, p.Logprob, p.Top = a.ID, lp, a.Top
	return nil
}

// completionResponse is what is kept of a /completion response.
type completionResponse struct {
	Content            string          `json:"content"`
	Tokens             []int           `json:"tokens"`
	StopType           string          `json:"stop_type"`
	Truncated          bool            `json:"truncated"`
	TokensEvaluated    int             `json:"tokens_evaluated"`
	TokensPredicted    int             `json:"tokens_predicted"`
	Probs              []tokenProb     `json:"completion_probabilities"`
	GenerationSettings json.RawMessage `json:"generation_settings"`
}

func (c *client) completion(ctx context.Context, req []byte) (completionResponse, error) {
	var r completionResponse
	err := c.do(ctx, http.MethodPost, "/completion", req, &r)
	return r, err
}
