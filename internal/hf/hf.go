// Package hf fetches files and byte ranges from Hugging Face Hub repos at a
// pinned revision.
package hf

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// Client downloads from huggingface.co. The zero value is not usable; use
// NewClient.
type Client struct {
	HTTP    *http.Client
	BaseURL string
	Token   string
}

// NewClient returns a client that authenticates with $HF_TOKEN if set.
func NewClient() *Client {
	return &Client{
		HTTP:    &http.Client{Timeout: 5 * time.Minute},
		BaseURL: "https://huggingface.co",
		Token:   os.Getenv("HF_TOKEN"),
	}
}

// URL returns the resolve URL of path in repo at revision.
func (c *Client) URL(repo, revision, path string) string {
	return fmt.Sprintf("%s/%s/resolve/%s/%s", c.BaseURL, repo, revision, path)
}

func (c *Client) get(ctx context.Context, url, rng string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	if rng != "" {
		req.Header.Set("Range", rng)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	want := http.StatusOK
	if rng != "" {
		want = http.StatusPartialContent
	}
	if resp.StatusCode != want {
		resp.Body.Close()
		return nil, fmt.Errorf("hf: GET %s (%s): %s", url, rng, resp.Status)
	}
	return resp, nil
}

// Fetch downloads a whole file.
func (c *Client) Fetch(ctx context.Context, repo, revision, path string) ([]byte, error) {
	resp, err := c.get(ctx, c.URL(repo, revision, path), "")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

// Sibling is a file entry of a repo revision.
type Sibling struct {
	Name   string `json:"rfilename"`
	BlobID string `json:"blobId"`
	Size   int64  `json:"size"`
	LFS    *struct {
		SHA256 string `json:"sha256"`
		Size   int64  `json:"size"`
	} `json:"lfs"`
}

// RepoInfo is the subset of the model info API used here.
type RepoInfo struct {
	SHA      string    `json:"sha"`
	Siblings []Sibling `json:"siblings"`
}

// Info returns the file listing of repo at revision, including LFS hashes.
func (c *Client) Info(ctx context.Context, repo, revision string) (*RepoInfo, error) {
	resp, err := c.get(ctx, fmt.Sprintf("%s/api/models/%s/revision/%s?blobs=true", c.BaseURL, repo, revision), "")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var info RepoInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("hf: decode repo info: %w", err)
	}
	return &info, nil
}

// RangeReader reads a remote file through HTTP range requests. It
// implements io.ReaderAt so safetensors.ReadHeader can use it directly.
type RangeReader struct {
	ctx    context.Context
	client *Client
	url    string
	// Size is the total file size from the last Content-Range header.
	Size int64
}

// Open returns a RangeReader for path in repo at revision.
func (c *Client) Open(ctx context.Context, repo, revision, path string) *RangeReader {
	return &RangeReader{ctx: ctx, client: c, url: c.URL(repo, revision, path), Size: -1}
}

// ReadAt fetches exactly len(p) bytes starting at off.
func (r *RangeReader) ReadAt(p []byte, off int64) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	resp, err := r.client.get(r.ctx, r.url, fmt.Sprintf("bytes=%d-%d", off, off+int64(len(p))-1))
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	var start, end, total int64
	if _, err := fmt.Sscanf(resp.Header.Get("Content-Range"), "bytes %d-%d/%d", &start, &end, &total); err == nil {
		if start != off {
			return 0, fmt.Errorf("hf: %s: asked for offset %d, got %d", r.url, off, start)
		}
		r.Size = total
	}
	return io.ReadFull(resp.Body, p)
}
