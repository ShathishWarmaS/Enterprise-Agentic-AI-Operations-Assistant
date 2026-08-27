"""Generate sample_data/capacity_guidelines.pdf so the repo ships a real PDF to
ingest without committing a binary blob that can't be reviewed. Run from repo root:

    python scripts/make_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

PAGES = [
    (
        "Capacity Guidelines",
        """These guidelines cover capacity planning for the payments platform.

payment-service runs 6 pods in steady state. Each pod holds a database
connection pool of 40 connections, for a total of 240. Peak throughput observed
is about 1.3 million requests per day.

Scale payment-service to 10 pods when error_rate_pct exceeds 2 percent and the
connection pool is saturated and there is no recent deploy to roll back. Scaling
adds pool capacity and is a temporary mitigation, not a fix.

Do not scale above 12 pods without a database review. Beyond 12 pods the shared
database becomes the bottleneck and adding pods increases lock contention.""",
    ),
    (
        "Redis capacity",
        """The merchant configuration cache uses a single Redis instance sized at
4 GB. Alert at 80 percent memory. The cache TTL is 15 minutes so memory use
should be bounded; sustained growth indicates a key that is being written
without a TTL and should be investigated.

Never run FLUSHALL against the shared Redis during business hours. Drop
individual keys with cachectl instead.""",
    ),
]


def main() -> None:
    out = Path("sample_data/capacity_guidelines.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for title, body in PAGES:
        page = doc.new_page()
        page.insert_text((72, 72), title, fontsize=18, fontname="helv")
        page.insert_textbox(fitz.Rect(72, 110, 523, 740), body, fontsize=11, fontname="helv")
    doc.save(out)
    doc.close()
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
