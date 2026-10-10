"""
evaluation/paired_stats.py
Paired diagnostic and statistical test entry point.
Exposes evaluation.paired_diagnostic functionality under the paired_stats name.
"""
from __future__ import annotations

import sys
from evaluation.paired_diagnostic import main, run, self_test, bg_mad, detect_parent_part

if __name__ == "__main__":
    sys.exit(main())
