"""Build N-layer Needle 3 rungs with the public exporter (W4A8).

The published needle3.cact is 2-bit, but the public exporter only packs 4-bit,
so every rung here (including a re-exported 20L) is W4 for a like-for-like sweep.
"""
import sys, argparse
import needle.model.export as E
E.read_layers = lambda path: -1  # never short-circuit to copying the published archive
from needle.model.finetune import build_main
for L in map(int, sys.argv[1:]):
    build_main(argparse.Namespace(checkpoint=None, lora=None, out=f"models/needle3-{L}L-w4.cact",
                                  upload=False, layers=L, platform=None))
