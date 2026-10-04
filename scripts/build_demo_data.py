"""Build the bundled demo corpus.

Generates a small, self-contained set of study material that exercises every
loader and every agent:

* ``thermodynamics_notes.pdf``  - a real PDF (multi-page, headings) via fpdf2
* ``thermodynamics_notes.md``   - the same content as Markdown
* ``cell_biology_review.csv``   - a table, loaded and preserved as a table
* ``lab_report.html``           - an HTML page with headings and a table
* ``molecules.txt``             - plain text

Run::

    .\\.venv\\Scripts\\python.exe scripts\\build_demo_data.py
    .\\.venv\\Scripts\\python.exe scripts\\build_demo_data.py --ingest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workcompanion.config.logging_config import configure_logging  # noqa: E402

configure_logging(level="WARNING", force=True)

DEMO_DIR = ROOT / "demo"

# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
#: ``(heading, body)`` pairs shared by the Markdown and PDF renderings.
SECTIONS: list[tuple[str, str]] = [
    (
        "Thermodynamics: Entropy and the Second Law",
        "Thermodynamics relates pressure, volume, temperature and energy. "
        "Entropy is the quantity that decides which processes can happen on "
        "their own.\n\n"
        "Entropy measures how many microstates are compatible with a "
        "macrostate. For an ideal gas:\n"
        "    S = nR ln(V/n) + n Cv ln(T) + constant\n"
        "Entropy is a state function, so the change between two states does "
        "not depend on the path taken.",
    ),
    (
        "The Second Law",
        "The second law of thermodynamics states that the entropy of an "
        "isolated system never decreases. A spontaneous process is one for "
        "which the total entropy change is positive.\n\n"
        "Three equivalent statements are used in practice:\n"
        "1. The entropy of an isolated system never decreases.\n"
        "2. Heat flows spontaneously from hot to cold, never the reverse.\n"
        "3. No cyclic device converts heat entirely into work.\n\n"
        "Entropy production is the irreversible part of a real process. "
        "Friction, diffusion and electrical resistance all produce entropy.",
    ),
    (
        "Gibbs Free Energy",
        "At constant temperature and pressure, spontaneity is decided by the "
        "Gibbs free energy:\n"
        "    G = H - T S\n"
        "A process is spontaneous when the change in G is negative. A process "
        "at equilibrium has dG equal to zero. A positive dG means the reverse "
        "reaction is spontaneous.\n\n"
        "G is only defined at constant temperature and pressure. Outside those "
        "conditions the correct potential is the Helmholtz free energy "
        "A = U - T S.",
    ),
    (
        "Enthalpy and Phase Changes",
        "Enthalpy H absorbs the energy of a phase change without a "
        "temperature change. This energy is latent heat.\n\n"
        "Latent heat is released in the reverse direction. At the melting "
        "point, adding heat raises the enthalpy but not the temperature until "
        "the phase change is complete.\n\n"
        "The latent heat of fusion of water at 0 degrees Celsius is "
        "approximately 334 kJ/kg. The latent heat of vaporisation at 100 "
        "degrees Celsius is approximately 2260 kJ/kg.",
    ),
    (
        "Entropy in Practice",
        "Refrigerators work because they do not violate the second law. They "
        "move heat from a cold reservoir to a hot one, but they consume work "
        "and therefore increase the total entropy.\n\n"
        "The Carnot efficiency bounds every heat engine:\n"
        "    eta = 1 - Tc / Th\n"
        "No engine operating between two reservoirs can exceed it, which is "
        "why practical engines fall well short of 100 percent.",
    ),
]

CELL_ROWS = [
    ("Eukaryotic cell", "Nucleus, mitochondria, ER", "100", "compartmentalised metabolism"),
    ("Prokaryotic cell", "Nucleoid, ribosomes", "1", "no membrane-bound organelles"),
    ("Ribosome", "rRNA and proteins", "20-30", "translation of mRNA into protein"),
    ("Mitochondrion", "Inner and outer membrane", "0.5-1", "oxidative phosphorylation"),
    ("Chloroplast", "Thylakoid and stroma", "5", "light-dependent photosynthesis"),
    ("Golgi apparatus", "Stacked cisternae", "1", "modifies and sorts proteins"),
    ("Lysosome", "Single membrane, acidic interior", "0.1-1.2", "intracellular digestion"),
    ("Nucleus", "Nuclear envelope and nucleolus", "1", "stores and replicates DNA"),
]

HTML_DOC = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Photosynthesis: a lab report</title>
</head>
<body>
  <h1>Photosynthesis: a lab report</h1>
  <p class="meta">Prepared for the WorkCompanion AI demo corpus.</p>

  <h2>Aim</h2>
  <p>Measure the rate of oxygen production in Elodea over six minutes and
  test whether the rate falls linearly as the light intensity is reduced.</p>

  <h2>Method</h2>
  <p>Elodea sprigs were placed in sodium hydrogen carbonate solution to supply
  carbon dioxide. A count of oxygen bubbles was taken every sixty seconds under
  each of four lamp distances.</p>

  <h2>Results</h2>
  <table border="1">
    <tr><th>Lamp distance (cm)</th><th>Bubbles at 6 min</th><th>Relative rate</th></tr>
    <tr><td>10</td><td>142</td><td>1.00</td></tr>
    <tr><td>25</td><td>96</td><td>0.68</td></tr>
    <tr><td>40</td><td>51</td><td>0.36</td></tr>
    <tr><td>60</td><td>24</td><td>0.17</td></tr>
  </table>

  <h2>Conclusion</h2>
  <p>Rate was not proportional to the inverse square of distance. The
  relationship flattened at close range, which suggests light saturation: once
  the photosystems are working at capacity, extra photons do not increase the
  rate of carbon fixation.</p>

  <h2>Limitations</h2>
  <p>Bubbles were counted by eye, temperature was not controlled, and each
  condition was measured only once. The experiment should be repeated with a
  controlled water bath and automated gas measurement.</p>
</body>
</html>
"""

TXT_DOC = """Quantum mechanics: superposition and measurement
=============================================

A quantum system is described by a wave function. Before measurement the
system is not in a definite classical state; it is in a superposition of the
possible outcomes.

Superposition
-------------
If |psi> is a state, then a|psi> + b|phi> is also a state, where a and b are
complex numbers. The only thing that matters physically is the ratio of a to
b, so the state lives on a ray rather than at a point.

Measurement
-----------
On measurement, one classical outcome is obtained. The probabilities of the
outcomes are the squared moduli |a|^2 and |b|^2. The wave function is not a
physical object: it cannot be observed directly, only through the statistics of
many measurements.

The measurement problem
-----------------------
Unitary evolution is deterministic and reversible. Measurement appears random
and irreversible. Reconciling the two without a preferred classical state is
the content of the measurement problem.

Applications
------------
Interference between two paths is the basis of the double-slit experiment and
of quantum computation, where interference between many amplitudes is used to
arrange for useful outcomes.
"""

LICENSE_NOTE = """WorkCompanion AI demo corpus.

These documents are short, synthetic teaching material written for this project.
They contain no copyrighted text.
"""


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_markdown(path: Path) -> Path:
    """The thermodynamics notes as Markdown."""
    lines = ["# Thermodynamics: notes", ""]
    for index, (heading, body) in enumerate(SECTIONS, start=1):
        if index > 1:
            lines += ["---", ""]
        lines += [f"## {heading}", "", body, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_text(path: Path) -> Path:
    path.write_text(TXT_DOC, encoding="utf-8")
    return path


def write_csv(path: Path) -> Path:
    """A table, so the CSV loader's table preservation is exercised."""
    lines = ["organelle,structure,size_um,function"]
    lines += [",".join(row) for row in CELL_ROWS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_html(path: Path) -> Path:
    path.write_text(HTML_DOC, encoding="utf-8")
    return path


def write_pdf(path: Path) -> Path:
    """A real multi-page PDF, so the PDF loader and page metadata are exercised."""
    try:
        from fpdf import FPDF
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "fpdf2 is required to build the demo PDF: pip install fpdf2"
        ) from exc

    class Notes(FPDF):
        def header(self) -> None:
            self.set_font("Helvetica", "I", 8)
            self.cell(self.epw, 8, "WorkCompanion AI - demo material", align="C")
            self.ln(12)

        def footer(self) -> None:
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.cell(self.epw, 10, f"Page {self.page_no()}", align="C")

    pdf = Notes(format="A4")
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(True, margin=18)
    pdf.set_title("Thermodynamics: Entropy and the Second Law")
    pdf.set_author("WorkCompanion AI demo corpus")

    def write(
        text: str, *, size: float = 10, font: str = "Helvetica", style: str = "", indent: float = 0.0
    ) -> None:
        """One wrapped paragraph.

        ``w`` is always the printable width rather than ``0``: fpdf2 computes
        ``w=0`` from the current x, and a header ``cell`` can leave x past the
        right margin, which then fails with "not enough horizontal space".
        """
        pdf.set_font(font, style, size)
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(pdf.epw - indent, size * 1.45, text)
        pdf.ln(2)

    pdf.add_page()
    write(
        "Thermodynamics: Entropy and the Second Law",
        size=17,
        font="Helvetica",
        style="B",
    )
    write(
        "A short set of teaching notes. They are deliberately factual and "
        "contain no copyrighted text."
    )

    for heading, body in SECTIONS[1:]:
        pdf.add_page()
        write(heading, size=14, font="Helvetica", style="B")
        for paragraph in body.split("\n\n"):
            for line in paragraph.splitlines():
                if line.startswith("    "):
                    write(line.strip(), font="Courier", indent=8)
                else:
                    write(line)

    pdf.output(str(path))
    return path


def write_readme(path: Path, files: list[Path]) -> Path:
    rows = "\n".join(
        f"| `{item.name}` | {item.suffix.lstrip('.').upper()} | {item.stat().st_size:,} |"
        for item in files
    )
    path.write_text(
        "# Demo corpus\n\n"
        "Synthetic teaching material for exercising every loader and agent.\n\n"
        "| File | Type | Bytes |\n| --- | --- | --- |\n"
        f"{rows}\n\n{LICENSE_NOTE}",
        encoding="utf-8",
    )
    return path


def build(target: Path = DEMO_DIR) -> list[Path]:
    """Write the whole corpus and return the files, in load order."""
    target.mkdir(parents=True, exist_ok=True)
    files = [
        write_markdown(target / "thermodynamics_notes.md"),
        write_pdf(target / "thermodynamics_notes.pdf"),
        write_csv(target / "cell_organelles.csv"),
        write_html(target / "photosynthesis_lab_report.html"),
        write_text(target / "quantum_mechanics.txt"),
    ]
    files.append(write_readme(target / "README.md", files))
    return files


def ingest(files: list[Path]) -> None:
    """Index the corpus into the configured store."""
    from workcompanion.agents.bundle import get_bundle
    from workcompanion.config.settings import get_settings

    settings = get_settings()
    bundle = get_bundle(settings)
    total = 0
    for path in files:
        if path.suffix.lower() == ".md" and path.name == "README.md":
            continue
        result = bundle.ingestion.ingest_file(
            path, subject=_subject_for(path.name), allow_duplicate=False
        )
        total += result.chunk_count
        mark = "ok " if result.ok else "FAIL"
        print(f"  [{mark}] {result.summary()}")
        if result.error:
            print(f"         error: {result.error}")
    bundle.nexus.set_document_state(not bundle.pipeline.is_empty)
    print(f"\nIndexed {total} chunk(s). The knowledge base is ready.")


def _subject_for(name: str) -> str:
    lowered = name.lower()
    if "thermo" in lowered:
        return "Thermodynamics"
    if "cell" in lowered:
        return "Biology"
    if "photo" in lowered:
        return "Biology"
    if "quantum" in lowered:
        return "Physics"
    return "General"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="index the corpus after building it",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEMO_DIR,
        help=f"output directory (default: {DEMO_DIR})",
    )
    args = parser.parse_args()

    print(f"Building demo corpus in {args.target} …")
    files = build(args.target)
    for item in files:
        print(f"  {item.stat().st_size:>9,}  {item.name}")
    print(f"\nWrote {len(files)} file(s) to {args.target}")
    if args.ingest:
        print()
        ingest(files)
    return 0


if __name__ == "__main__":
    sys.exit(main())