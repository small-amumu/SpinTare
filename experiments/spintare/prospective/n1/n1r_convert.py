# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""Run the census convert_hf_to_gguf.py with a pre-tokenizer mapping for two N1 models.

The census converter (llama.cpp 3bb63cb) does not list the chkhsh of DeepSeek-R1-Distill-Llama-8B and
DeepSeek-R1-0528-Qwen3-8B. Their tokenizer.json has the same pre_tokenizer, normalizer and base vocabulary as
Meta-Llama-3.1-8B-Instruct and Qwen3-8B, whose census GGUFs carry tokenizer.ggml.pre llama-bpe and qwen2; the
hashes differ through the added special tokens only. This wrapper returns that name for exactly these two hashes
and leaves every other path of the converter unchanged (any other unknown hash still raises).
Usage: python n1r_convert.py <census llama.cpp dir> <convert_hf_to_gguf.py args ...>
"""
from __future__ import annotations

import runpy
import sys
from hashlib import sha256
from pathlib import Path

MAP = {
    "a63e0d2191069e2b760b155e39a0d9a348ec29a6ee47e8beb24a1174a4b8f197": "llama-bpe",  # R1-Distill-Llama-8B
    "0d75215efe33c49084836cb245f2fa78de4b3858f5a3e54d5e1fd27f4ce33b05": "qwen2",      # R1-0528-Qwen3-8B
}


def main() -> None:
    conv = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(conv))
    sys.path.insert(1, str(conv / "gguf-py"))  # as convert_hf_to_gguf.py does (review r48 M-3)
    import gguf  # noqa: E402
    import conversion.base as base  # noqa: E402  (the census converter's own module)

    print(f"n1r_convert: gguf from {gguf.__file__}", file=sys.stderr)
    assert Path(gguf.__file__).resolve().is_relative_to(conv / "gguf-py"), "gguf is not the census gguf-py"

    orig = base.TextModel.get_vocab_base_pre
    chktxt = [c for c in orig.__code__.co_consts if isinstance(c, str) and c.startswith("\n \n\n \n\n\n")]
    assert len(chktxt) == 1, "census chktxt not found"

    def patched(self, tokenizer) -> str:
        try:
            return orig(self, tokenizer)
        except NotImplementedError:
            h = sha256(str(tokenizer.encode(chktxt[0])).encode()).hexdigest()
            if h in MAP:
                print(f"n1r_convert: chkhsh {h} -> {MAP[h]}", file=sys.stderr)
                return MAP[h]
            raise

    base.TextModel.get_vocab_base_pre = patched
    sys.argv = [str(conv / "convert_hf_to_gguf.py")] + sys.argv[2:]
    runpy.run_path(str(conv / "convert_hf_to_gguf.py"), run_name="__main__")


if __name__ == "__main__":
    main()
