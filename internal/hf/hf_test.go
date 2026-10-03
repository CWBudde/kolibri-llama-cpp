package hf

import (
	"bytes"
	"context"
	"encoding/binary"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"kolibri-llm/internal/safetensors"
)

func TestRangeReaderReadsSafetensorsHeader(t *testing.T) {
	hdr := `{"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}`
	var file bytes.Buffer
	_ = binary.Write(&file, binary.LittleEndian, uint64(len(hdr)))
	file.WriteString(hdr)
	file.Write(make([]byte, 12))

	var ranges []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/org/model/resolve/abc/model.safetensors" {
			http.NotFound(w, r)
			return
		}
		ranges = append(ranges, r.Header.Get("Range"))
		http.ServeContent(w, r, "", time.Time{}, bytes.NewReader(file.Bytes()))
	}))
	defer srv.Close()

	c := NewClient()
	c.BaseURL, c.Token = srv.URL, ""
	r := c.Open(context.Background(), "org/model", "abc", "model.safetensors")
	h, err := safetensors.ReadHeader(r)
	if err != nil {
		t.Fatal(err)
	}
	if len(h.Tensors) != 1 || h.Tensors[0].Name != "w" {
		t.Errorf("tensors = %+v", h.Tensors)
	}
	if r.Size != int64(file.Len()) || h.FileSize() != r.Size {
		t.Errorf("Size = %d, header implies %d, file is %d", r.Size, h.FileSize(), file.Len())
	}
	want := []string{"bytes=0-7", "bytes=8-" + strconv.Itoa(8+len(hdr)-1)}
	if len(ranges) != 2 || ranges[0] != want[0] || ranges[1] != want[1] {
		t.Errorf("ranges = %v, want %v", ranges, want)
	}

	if _, err := c.Open(context.Background(), "org/model", "abc", "missing").ReadAt(make([]byte, 8), 0); err == nil {
		t.Error("404 accepted")
	}
}
