"""Single driver: regenerate every figure used in the contribution docs.

Calls the two per-PR figure modules in sequence. Equivalent to running
``figures/figures_icl.py`` then ``figures/figures_hddc.py``.

Run from the repo root or this folder:

    python figures/make_all.py
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures_icl                                              # noqa: E402
import figures_hddc                                             # noqa: E402

log = logging.getLogger("make_all")


def main() -> None:
    figures_icl.main()
    figures_hddc.main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
