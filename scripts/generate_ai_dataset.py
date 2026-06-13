#!/usr/bin/env python3
"""Generate an AI-regenerated dataset paired 1:1 with a source public/<dataset> tree.

Thin entry point — the implementation lives in the ``ai_dataset`` package next to this
file. See ``ai_dataset/__init__.py`` for the architecture and module layout.
"""
from ai_dataset.cli import main

if __name__ == "__main__":
    main()
