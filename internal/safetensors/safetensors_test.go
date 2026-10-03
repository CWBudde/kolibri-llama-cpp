package safetensors

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"
)

func file(header string, dataLen int) []byte {
	var b bytes.Buffer
	_ = binary.Write(&b, binary.LittleEndian, uint64(len(header)))
	b.WriteString(header)
	b.Write(make([]byte, dataLen))
	return b.Bytes()
}

func TestReadHeader(t *testing.T) {
	hdr := `{"__metadata__":{"format":"pt"},` +
		`"b":{"dtype":"F32","shape":[3],"data_offsets":[8,20]},` +
		`"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},` +
		`"s":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[20,24]}}`
	h, err := ReadHeader(bytes.NewReader(file(hdr, 24)))
	if err != nil {
		t.Fatal(err)
	}
	if h.Metadata["format"] != "pt" {
		t.Errorf("metadata = %v", h.Metadata)
	}
	var names []string
	for _, tt := range h.Tensors {
		names = append(names, tt.Name)
	}
	if got := strings.Join(names, ","); got != "a,b,s" {
		t.Errorf("tensors not sorted by offset: %s", got)
	}
	if h.DataSize() != 24 || h.FileSize() != int64(8+len(hdr)+24) {
		t.Errorf("DataSize=%d FileSize=%d", h.DataSize(), h.FileSize())
	}
	if n := h.Tensors[0].NumElements(); n != 4 {
		t.Errorf("NumElements = %d", n)
	}
}

func TestParseHeaderRejects(t *testing.T) {
	tests := []struct {
		name, header, want string
	}{
		{"size mismatch", `{"a":{"dtype":"BF16","shape":[3],"data_offsets":[0,8]}}`, "want 6"},
		{"gap", `{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}`, "starts at 1"},
		{"overlap", `{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}`, "starts at 1"},
		{"dtype", `{"a":{"dtype":"Q4","shape":[1],"data_offsets":[0,1]}}`, "unknown dtype"},
		{"json", `{"a":`, "decode header"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseHeader([]byte(tc.header))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want containing %q", err, tc.want)
			}
		})
	}
}

func TestReadHeaderSize(t *testing.T) {
	if _, err := ReadHeaderSize([]byte{1, 2}); err == nil {
		t.Error("short prefix accepted")
	}
	big := make([]byte, 8)
	binary.LittleEndian.PutUint64(big, MaxHeaderSize+1)
	if _, err := ReadHeaderSize(big); err == nil {
		t.Error("oversized header accepted")
	}
}
