#!/usr/bin/env python3
"""Check source integrity and run the implementation tests."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parent

def verify_sources():
    for path in ROOT.rglob("*.py"):
        if not any(part in (".venv", "runs", "__pycache__") for part in path.parts):
            compile(path.read_text(), str(path), "exec")
    hashes = json.loads((ROOT / "third_party/source_hashes.json").read_text())
    for name, expected in hashes["implementation_sha256"].items():
        if hashlib.sha256((ROOT / "fscil" / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Implementation checksum mismatch: " + name)
    vocabulary = ROOT / "models/clip/bpe_simple_vocab_16e6.txt.gz"
    if hashlib.sha256(vocabulary.read_bytes()).hexdigest() != hashes["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer checksum mismatch")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syntax-only", action="store_true")
    args = parser.parse_args()
    verify_sources()
    print("Source syntax and implementation integrity: PASS", flush=True)
    if args.syntax_only:
        return
    suite = unittest.defaultTestLoader.discover(str(ROOT / "fscil/tests"), top_level_dir=str(ROOT))
    if not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful():
        raise SystemExit(1)

if __name__ == "__main__":
    main()
