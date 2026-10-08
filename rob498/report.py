"""Summary-table builders for the report deliverables.

The lab requires a specific summary table, and it is described as
"the centre of the report". These builders emit Markdown (paste into your
report) and CSV (for the graders), and they refuse to emit a row whose
conditions are missing -- which is the course's stated grading rule turned into
code.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


def _fmt(v: Any, places: int = 4) -> str:
    if v is None:
        return "--"
    if hasattr(v, "dtype") and getattr(v, "ndim", 1) == 0:   # NumPy scalar
        v = v.item()
    if isinstance(v, float):
        if v != v:                       # NaN
            return "n/a"
        if v == 0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.{places}g}"
        return f"{v:.{places}f}".rstrip("0").rstrip(".")
    return str(v)


@dataclass
class Table:
    """A results table that carries its conditions.

    The handout's example of a worthless row is ``VGGT: 0.03``; the example of a
    full-marks row names the sequence, the trial count, the seed, and the
    hardware. ``conditions`` is where that goes, and :meth:`to_markdown` refuses
    to render without it.

    Example:
        t = Table("Lab 1 summary", ["metric", "value"])
        t.add_row(["3DGS PSNR (dB)", 27.4])
        t.add_row(["3DGS accuracy at 5 cm (m)", 0.031])
        t.conditions = manifest.conditions_line()
        print(t.to_markdown())
    """

    title: str
    columns: Sequence[str]
    rows: list[list[Any]] = field(default_factory=list)
    conditions: str = ""
    notes: str = ""

    def add_row(self, row: Sequence[Any]) -> None:
        if len(row) != len(self.columns):
            raise ValueError(f"row has {len(row)} entries, expected {len(self.columns)}")
        self.rows.append(list(row))

    def to_markdown(self, strict: bool = True) -> str:
        if strict and not self.conditions:
            raise ValueError(
                "Table.conditions is empty. Every quantitative claim in this course "
                "needs its conditions attached (sequence, trials, seed, hardware). "
                "Set table.conditions, e.g. from RunManifest.conditions_line(). "
                "Pass strict=False only for a table that genuinely has no run "
                "conditions, such as a memory-accounting table."
            )
        head = "| " + " | ".join(self.columns) + " |"
        sep = "|" + "|".join("---" for _ in self.columns) + "|"
        body = ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in self.rows]
        out = [f"**{self.title}**", "", head, sep, *body]
        if self.conditions:
            out += ["", f"_Conditions: {self.conditions}_"]
        if self.notes:
            out += ["", self.notes]
        return "\n".join(out)

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.columns)
            w.writerows(self.rows)
            if self.conditions:
                w.writerow([])
                w.writerow(["# conditions", self.conditions])
        return path


# --------------------------------------------------------------------------- #
# The required summary tables, pre-shaped
# --------------------------------------------------------------------------- #
def lab1_summary_table() -> Table:
    """Lab 1's central table: representations as rows.

    Deliverable: "representations as rows and columns for fidelity, memory,
    build time, query latency, and your task metric."

    The ordering is the point of the lab. If the fidelity column and the task
    column rank the three rows the same way, you have probably not found the
    flip that Part E3 asks for -- look harder before concluding it does not
    exist.
    """
    t = Table(
        "Lab 1 -- representation comparison",
        ["representation", "fidelity", "memory (MB)", "build time (s)",
         "query latency (ms)", "task metric", "task FN rate"],
    )
    for rep in ["Learned segmentation (B)", "3D Gaussian Splatting (C)",
                "Feed-forward / VGGT (D)", "TSDF baseline (E2)"]:
        t.add_row([rep, None, None, None, None, None, None])
    t.notes = ("State in the caption which fidelity metric the column holds "
               "(PSNR? accuracy? f-score?) -- they rank differently, and the "
               "reader cannot tell from a bare number.")
    return t


def write_report_bundle(
    out_dir: str | Path, tables: Sequence[Table], strict: bool = True
) -> Path:
    """Write every table to one Markdown file plus one CSV each.

    Paste the Markdown into your report; the CSVs go in the code archive so a
    grader can diff your numbers without reading your PDF.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_parts = []
    for i, t in enumerate(tables):
        md_parts.append(t.to_markdown(strict=strict))
        slug = "".join(ch if ch.isalnum() else "_" for ch in t.title.lower())[:60]
        t.to_csv(out_dir / f"{i:02d}_{slug}.csv")
    md = out_dir / "tables.md"
    md.write_text("\n\n---\n\n".join(md_parts) + "\n")
    return md
