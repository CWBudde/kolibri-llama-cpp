#!/usr/bin/env python3
"""Write a vocab-only Kolibri-1 GGUF with llama.cpp's own converter code.

Phase 2 shim: libllama has no model class for the kolibri architecture yet
(Phase 4), and llama_model_create rejects it even with vocab_only=true. The
vocabulary is therefore written under the placeholder architecture qwen3moe.
llama.cpp skips all hyperparameters when it loads a model with
vocab_only=true, so the placeholder does not influence tokenization. Once
llama_model_kolibri exists, convert_hf_to_gguf.py --vocab-only replaces this.

    vocab_gguf.py --llama-cpp third_party/llama.cpp --out third_party/llama.cpp/models/ggml-vocab-kolibri.gguf

Without --tokenizer-dir it uses the pinned BF16 revision from the Phase 1
inventory, checked against the recorded sha256.
"""

import argparse
import sys
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout with the kolibri pre-tokenizer patch")
    ap.add_argument("--tokenizer-dir", type=Path, help="directory with config.json, tokenizer.json, tokenizer_config.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.tokenizer_dir is None:
        from common import tokenizer_dir
        args.tokenizer_dir = tokenizer_dir()

    sys.path.insert(0, str(args.llama_cpp / "gguf-py"))
    sys.path.insert(0, str(args.llama_cpp))
    import gguf
    from conversion import ModelBase, TextModel

    @ModelBase.register("Kolibri1ForCausalLM")
    class KolibriVocab(TextModel):
        model_arch = gguf.MODEL_ARCH.QWEN3MOE  # placeholder, see module docstring

        def set_vocab(self):
            self._set_vocab_gpt2()

        def set_gguf_parameters(self):
            pass  # vocab-only: llama.cpp does not read hyperparameters

    model = KolibriVocab(args.tokenizer_dir, gguf.LlamaFileType.ALL_F32, args.out, model_name="Kolibri-1")
    with tempfile.TemporaryDirectory() as empty:
        # The HF cache directory is named after the revision SHA, which the
        # metadata heuristics would otherwise record as general.finetune.
        model.dir_model_card = Path(empty)
        model.write_vocab()


if __name__ == "__main__":
    main()
