// Package memctl controls memory for the expert cache benchmark on macOS,
// without root: it wires anonymous memory, a memory ceiling the compressor
// and swap cannot undo, and it counts and evicts a file's pages in the page
// cache, for cold runs. It is darwin only.
package memctl
