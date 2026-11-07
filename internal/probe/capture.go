package probe

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// CaptureSchema names capture.json's layout, which tools/probe/capture.py writes.
const CaptureSchema = "kolibri-probe-capture-v1"

// Capture is capture.json as far as the report checks it. The rest (model, server and template
// hashes, ...) is carried over whole into the report as Raw.
type Capture struct {
	Schema             string               `json:"schema"`
	DatasetSHA256      string               `json:"dataset_sha256"`
	LogitsCheck        string               `json:"logits_check"` // equal, differ or skipped
	Repeatable         *bool                `json:"repeatable"`   // null: no repeat ran
	NLayer             int                  `json:"n_layer"`
	GraphCheck         map[string]GraphNode `json:"graph_check"`
	LastTokens         []int                `json:"last_tokens"`
	TemplateIsReleased *bool                `json:"template_is_released"` // null: no released template to compare with
	Reasoning          string               `json:"reasoning"`
	Device             string               `json:"device"`
	NUbatchEffective   int                  `json:"n_ubatch_effective"`
	ExpertCacheSlots   int                  `json:"expert_cache_slots"`
	ModelSHA256        string               `json:"model_sha256"`
	Rows               []CaptureRow         `json:"rows"`
	Layers             []CaptureLayer       `json:"layers"`
	Raw                map[string]any       `json:"-"`
}

// GraphNode is what the capture read from the model's graph about one l_out node.
type GraphNode struct {
	Sources []string `json:"sources"`
	PostFFN PostFFN  `json:"post_ffn"`
	ReadBy  string   `json:"read_by"`
}

// PostFFN is the node's first source, from the add down: the norm's weight multiply, the RMS norm
// under it and the sum of the experts under that.
type PostFFN struct {
	Ops     []string `json:"ops"`
	Weight  string   `json:"weight"`
	Norm    string   `json:"norm"`
	Experts []string `json:"experts"`
}

// CaptureRow is one prompt as captured.
type CaptureRow struct {
	ID        string `json:"id"`
	NTokens   int    `json:"n_tokens"`
	LastToken int    `json:"last_token"`
}

// CaptureLayer is one captured depth. OneBased is the issue's numbering (1 to n_layer), ZeroBased
// the graph's il, and Node the tensor read: l_out-<il>.
type CaptureLayer struct {
	OneBased  int    `json:"one_based"`
	ZeroBased int    `json:"zero_based"`
	Node      string `json:"node"`
	File      string `json:"file"`
	SHA256    string `json:"sha256"`
	Dtype     string `json:"dtype"`
	Shape     [2]int `json:"shape"`
}

// LayerActivations is one layer read from a capture.
type LayerActivations struct {
	Layer  CaptureLayer
	Matrix Matrix
}

// checkGraph refuses a layer whose node is not wired as docs/layer-probe.md says: the residual add
// of ffn_inp and post_ffw_norm(routed + shared experts), read by the next block's norm or output_norm.
func (c Capture) checkGraph(l CaptureLayer) error {
	il := l.ZeroBased
	g, ok := c.GraphCheck[l.Node]
	if !ok {
		return fmt.Errorf("capture.json: no graph_check for %s: its wiring was not checked", l.Node)
	}
	readBy := fmt.Sprintf("norm-%d", il+1)
	if il+1 == c.NLayer {
		readBy = "norm"
	}
	if len(g.Sources) != 2 || g.Sources[1] != fmt.Sprintf("ffn_inp-%d", il) || g.ReadBy != readBy ||
		!slices.Equal(g.PostFFN.Ops, []string{"MUL", "RMS_NORM", "ADD"}) ||
		g.PostFFN.Weight != fmt.Sprintf("blk.%d.post_ffw_norm.weight", il) ||
		g.PostFFN.Norm != fmt.Sprintf("norm-%d", il) ||
		!slices.Equal(g.PostFFN.Experts, []string{fmt.Sprintf("ffn_moe_out-%d", il), fmt.Sprintf("ffn_shexp_out-%d", il)}) {
		return fmt.Errorf("capture.json: %s is wired as %+v, not as the residual add of ffn_inp-%d and the post-FFN norm, read by %s", l.Node, g, il, readBy)
	}
	return nil
}

// Facts are what a reader of the report needs to know the capture was checked against.
func (c Capture) Facts() []string {
	released := "not known: no released template to compare with"
	if c.TemplateIsReleased != nil {
		released = "yes"
		if !*c.TemplateIsReleased {
			released = "**no**"
		}
	}
	nodes := make([]string, 0, len(c.Layers))
	for _, l := range c.Layers {
		nodes = append(nodes, l.Node)
	}
	return []string{
		"Logits against a context without a callback: " + c.LogitsCheck + ".",
		"Repeated decode: equal.",
		fmt.Sprintf("Graph, read from the model on the first decode: %s are each the residual add of ffn_inp and the post-FFN norm.", strings.Join(nodes, ", ")),
		fmt.Sprintf("Last prompt token, the same for every prompt: %d. Reasoning mode: %s.", c.LastTokens[0], c.Reasoning),
		"Chat template is the released one: " + released + ".",
		fmt.Sprintf("Device %s, effective ubatch %d, expert cache slots %d. Model sha256 %s.", c.Device, c.NUbatchEffective, c.ExpertCacheSlots, c.ModelSHA256),
	}
}

// ReadCapture reads a capture directory and refuses one that does not belong to the dataset's
// bytes, whose files differ from what capture.json records, whose row order is not the dataset's,
// whose capture changed the logits or was not compared with a plain run, whose repeated decode
// differed or did not run, whose graph is not wired as documented, or whose prompts do not all end
// in one token.
func ReadCapture(dir string, datasetBytes []byte, items []Item) (Capture, []LayerActivations, error) {
	var c Capture
	raw, err := os.ReadFile(filepath.Join(dir, "capture.json"))
	if err != nil {
		return c, nil, err
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return c, nil, fmt.Errorf("capture.json: %w", err)
	}
	if err := json.Unmarshal(raw, &c.Raw); err != nil {
		return c, nil, fmt.Errorf("capture.json: %w", err)
	}
	if c.Schema != CaptureSchema {
		return c, nil, fmt.Errorf("capture.json: schema %q, want %q", c.Schema, CaptureSchema)
	}
	if got := sha256Hex(datasetBytes); c.DatasetSHA256 != got {
		return c, nil, fmt.Errorf("capture read dataset %s, this one is %s", c.DatasetSHA256, got)
	}
	switch c.LogitsCheck {
	case "equal":
	case "skipped":
		return c, nil, fmt.Errorf("capture.json: logits_check skipped: nothing shows this capture left the logits alone; capture again without --skip-logits-check")
	default:
		return c, nil, fmt.Errorf("capture.json: logits_check %q: the capture changed the logits", c.LogitsCheck)
	}
	if c.Repeatable == nil {
		return c, nil, fmt.Errorf("capture.json: no repeated decode ran: nothing shows this capture is reproducible; capture again without --no-repeat")
	}
	if !*c.Repeatable {
		return c, nil, fmt.Errorf("capture.json: a repeated decode differed: the capture is not reproducible")
	}
	if len(c.LastTokens) != 1 {
		return c, nil, fmt.Errorf("capture.json: last_tokens %v: the prompts must all end in one token", c.LastTokens)
	}
	if len(c.Rows) != len(items) {
		return c, nil, fmt.Errorf("capture has %d rows, the dataset %d", len(c.Rows), len(items))
	}
	for i, r := range c.Rows {
		if r.ID != items[i].ID {
			return c, nil, fmt.Errorf("capture row %d is %s, the dataset's %s", i, r.ID, items[i].ID)
		}
		if r.LastToken != c.LastTokens[0] {
			return c, nil, fmt.Errorf("capture row %s ends in token %d, last_tokens says %d", r.ID, r.LastToken, c.LastTokens[0])
		}
	}
	if len(c.Layers) == 0 {
		return c, nil, fmt.Errorf("capture has no layers")
	}
	var out []LayerActivations
	for _, l := range c.Layers {
		if l.ZeroBased != l.OneBased-1 || l.Node != fmt.Sprintf("l_out-%d", l.ZeroBased) {
			return c, nil, fmt.Errorf("layer %d: zero-based %d, node %s: the numbering is not node l_out-(N-1)", l.OneBased, l.ZeroBased, l.Node)
		}
		if err := c.checkGraph(l); err != nil {
			return c, nil, err
		}
		data, err := os.ReadFile(filepath.Join(dir, l.File))
		if err != nil {
			return c, nil, err
		}
		if got := sha256Hex(data); got != l.SHA256 {
			return c, nil, fmt.Errorf("%s: sha256 %s, capture.json says %s", l.File, got, l.SHA256)
		}
		m, err := ReadNpy(bytes.NewReader(data))
		if err != nil {
			return c, nil, fmt.Errorf("%s: %w", l.File, err)
		}
		if l.Dtype != "float32" || m.Rows != l.Shape[0] || m.Dim != l.Shape[1] || m.Rows != len(items) {
			return c, nil, fmt.Errorf("%s: %s %dx%d, capture.json says %s %v, the dataset has %d rows", l.File, "float32", m.Rows, m.Dim, l.Dtype, l.Shape, len(items))
		}
		out = append(out, LayerActivations{Layer: l, Matrix: m})
	}
	return c, out, nil
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
