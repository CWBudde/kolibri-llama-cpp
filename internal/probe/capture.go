package probe

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// CaptureSchema names capture.json's layout, which tools/probe/capture.py writes.
const CaptureSchema = "kolibri-probe-capture-v1"

// Capture is capture.json as far as the report checks it. The rest (model, server and template
// hashes, ...) is carried over whole into the report as Raw.
type Capture struct {
	Schema        string         `json:"schema"`
	DatasetSHA256 string         `json:"dataset_sha256"`
	LogitsCheck   string         `json:"logits_check"` // equal or skipped
	Repeatable    *bool          `json:"repeatable"`
	Rows          []CaptureRow   `json:"rows"`
	Layers        []CaptureLayer `json:"layers"`
	Raw           map[string]any `json:"-"`
}

// CaptureRow is one prompt as captured.
type CaptureRow struct {
	ID      string `json:"id"`
	NTokens int    `json:"n_tokens"`
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

// ReadCapture reads a capture directory and refuses one that does not belong to the dataset's
// bytes, whose files differ from what capture.json records, whose row order is not the dataset's,
// or whose capture changed the logits.
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
	if c.LogitsCheck != "equal" && c.LogitsCheck != "skipped" {
		return c, nil, fmt.Errorf("capture.json: logits_check %q: the capture changed the logits", c.LogitsCheck)
	}
	if len(c.Rows) != len(items) {
		return c, nil, fmt.Errorf("capture has %d rows, the dataset %d", len(c.Rows), len(items))
	}
	for i, r := range c.Rows {
		if r.ID != items[i].ID {
			return c, nil, fmt.Errorf("capture row %d is %s, the dataset's %s", i, r.ID, items[i].ID)
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
