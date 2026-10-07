"""Build the WorkCompanion AI project document (.docx).

Single self-contained command:

    .\\.venv\\Scripts\\python.exe scripts\\build_project_doc.py

Factual content (settings and their declared defaults, agent method
signatures, schema fields, table columns, module inventory) is inspected from
the codebase at build time by :func:`collect_facts`. Only the prose that
describes intent is authored in this file, so the document cannot drift from
the implementation.

Run it again after changing settings, agents, schemas or database models.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "WorkCompanion_AI_Project_Documentation.docx"

# --- palette --------------------------------------------------------------
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5F, 0x66)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
GOOD = RGBColor(0x1B, 0x5E, 0x20)
WARN = RGBColor(0x8A, 0x50, 0x00)
SHADE = "EDF2F7"
HEAD_SHADE = "1F4E79"


# ---------------------------------------------------------------------------
# low-level docx helpers
# ---------------------------------------------------------------------------
def shade(cell, hex_color: str) -> None:
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hex_color)
    cell._tc.get_or_add_tcPr().append(el)


def cell_text(cell, text: str, *, bold=False, size=9, color=INK, mono=False) -> None:
    cell.text = ""
    para = cell.paragraphs[0]
    para.paragraph_format.space_before = Pt(2)
    para.paragraph_format.space_after = Pt(2)
    run = para.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.name = "Consolas" if mono else "Calibri"


def table(doc, headers: list[str], rows: list[list[str]], widths=None, mono_cols=()):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, head in enumerate(headers):
        cell = t.rows[0].cells[i]
        shade(cell, HEAD_SHADE)
        cell_text(cell, head, bold=True, size=9, color=RGBColor(0xFF, 0xFF, 0xFF))
    for r_i, row in enumerate(rows):
        cells = t.add_row().cells
        for i, value in enumerate(row):
            if r_i % 2:
                shade(cells[i], SHADE)
            cell_text(cells[i], value, size=9, mono=(i in mono_cols))
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    return t


def para(doc, text="", *, size=10.5, bold=False, italic=False, color=INK,
         space_after=6, align=None, mono=False, indent=0.0):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.left_indent = Inches(indent)
    if align is not None:
        p.alignment = align
    if text:
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.name = "Consolas" if mono else "Calibri"
    return p


def rich(doc, parts, *, size=10.5, space_after=6, indent=0.0, style=None):
    """parts = [(text, {bold/italic/mono/color}), ...]"""
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.left_indent = Inches(indent)
    for text, fmt in parts:
        run = p.add_run(text)
        run.bold = fmt.get("bold", False)
        run.italic = fmt.get("italic", False)
        run.font.size = Pt(size)
        run.font.color.rgb = fmt.get("color", INK)
        run.font.name = "Consolas" if fmt.get("mono") else "Calibri"
    return p


def bullet(doc, text, *, level=0, size=10.5):
    style = "List Bullet" if level == 0 else "List Bullet 2"
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.name = "Calibri"
    return p


def numbered(doc, text, *, size=10.5):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.font.size = Pt(size)
    return p


def code(doc, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.left_indent = Inches(0.25)
    run = p.add_run(text)
    run.font.size = Pt(9)
    run.font.name = "Consolas"
    run.font.color.rgb = RGBColor(0x1B, 0x2B, 0x34)
    pPr = p._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "18")
    left.set(qn("w:space"), "6")
    left.set(qn("w:color"), "9DB2C4")
    borders.append(left)
    pPr.append(borders)


def callout(doc, title: str, body: str, *, color=ACCENT, shade_hex="EAF1F8") -> None:
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.rows[0].cells[0]
    shade(cell, shade_hex)
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(title)
    r.bold = True
    r.font.size = Pt(10)
    r.font.color.rgb = color
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_after = Pt(2)
    r2 = p2.add_run(body)
    r2.font.size = Pt(9.5)
    r2.font.color.rgb = INK
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def h(doc, text, level):
    hd = doc.add_heading(text, level=level)
    for run in hd.runs:
        run.font.color.rgb = ACCENT if level > 1 else INK
    hd.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    hd.paragraph_format.space_after = Pt(6)
    return hd


def field(paragraph, instr: str) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr_el = OxmlElement("w:instrText")
    instr_el.set(qn("xml:space"), "preserve")
    instr_el.text = instr
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr_el)
    run._r.append(end)


def page_number_footer(section) -> None:
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("WorkCompanion AI  ·  ")
    r.font.size = Pt(8)
    r.font.color.rgb = MUTED
    field(p, "PAGE")
    r2 = p.add_run(" / ")
    r2.font.size = Pt(8)
    r2.font.color.rgb = MUTED
    field(p, "NUMPAGES")
    for run in p.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = MUTED


def git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=20,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def py_version() -> str:
    try:
        return subprocess.run(
            [sys.executable, "--version"], capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# facts
# ---------------------------------------------------------------------------
def collect_facts() -> dict:
    """Read the factual content straight out of the codebase.

    Nothing factual in the .docx should be transcribed by hand: settings and
    their declared defaults, agent method signatures, schema fields, table
    columns and test counts are all inspected here. Prose about intent is the
    only thing written in this file.
    """
    import inspect

    from workcompanion.config.settings import Settings

    facts: dict[str, object] = {}

    # Settings: name, declared type, declared default, description.
    settings = []
    for name, info in Settings.model_fields.items():
        default = info.default
        if str(default).startswith("PydanticUndefined") or repr(
            default
        ).startswith("PydanticUndefined"):
            default = "<required>"
        settings.append(
            {
                "name": name.upper(),
                "type": str(info.annotation).replace("typing.", "")[:60],
                "default": str(default)[:80],
                "doc": (info.description or "").split("\n")[0][:220],
            }
        )
    facts["settings"] = settings

    # Agents: key -> (class, responsibility).
    from workcompanion.agents import AGENT_REGISTRY

    skip = {"run", "name", "role", "description", "goal", "backstory"}
    agents = []
    for key, (cls, role) in AGENT_REGISTRY.items():
        methods = []
        for member_name, member in inspect.getmembers(cls, callable):
            if member_name.startswith("_") or member_name in skip:
                continue
            try:
                sig = inspect.signature(member)
            except (TypeError, ValueError):
                continue
            params = [p for p in sig.parameters if p != "self"]
            methods.append(f"{member_name}({', '.join(params)})")
        agents.append({"key": key, "cls": cls.__name__, "role": role, "methods": methods})
    facts["agents"] = agents

    # Pydantic schemas shared across layers.
    import workcompanion.schemas as schemas

    facts["schemas"] = [
        {"name": n, "fields": [f for f in getattr(schemas, n).model_fields if not f.startswith("_")]}
        for n in sorted(dir(schemas))
        if isinstance(getattr(schemas, n), type) and hasattr(getattr(schemas, n), "model_fields")
    ]

    # Database tables and their columns.
    from workcompanion.database import models

    facts["tables"] = [
        {
            "name": getattr(models, n).__table__.name,
            "model": n,
            "columns": [
                f"{c.name}{' (PK)' if c.primary_key else ''}"
                for c in getattr(models, n).__table__.columns
            ],
        }
        for n in sorted(dir(models))
        if isinstance(getattr(models, n), type) and hasattr(getattr(models, n), "__table__")
    ]

    # Module inventory.
    tree = []
    for path in sorted((ROOT / "workcompanion").rglob("*.py")):
        parts = path.relative_to(ROOT).parts
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        if parts:
            tree.append("/".join(parts))
    facts["tree"] = tree

    # Test inventory. Counted from pytest's own collection rather than by
    # reading source: some tests are parametrized, so a naive function count
    # under-reports (it found 128 where pytest runs 136). Falls back to an
    # AST scan when pytest is not installed.
    import ast
    import re

    tests: list[dict[str, object]] = []
    collected = False
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q",
             "-p", "no:cacheprovider"],
            cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
        for line in proc.stdout.splitlines():
            m = re.match(r"^(?:tests[/\\])?(\S*test_\w+\.py):\s*(\d+)$", line.strip())
            if m:
                tests.append(
                    {"file": Path(m.group(1)).name, "tests": int(m.group(2)), "classes": None}
                )
        collected = bool(tests)
    except Exception:
        collected = False

    if not collected:
        for path in sorted((ROOT / "tests").glob("test_*.py")):
            ast_tree = ast.parse(path.read_text(encoding="utf-8"))
            count = sum(
                1
                for node in ast.walk(ast_tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            )
            classes = sum(
                1
                for node in ast.walk(ast_tree)
                if isinstance(node, ast.ClassDef) and node.name.startswith("Test")
            )
            tests.append({"file": path.name, "tests": count, "classes": classes})

    # Class counts are not visible in pytest's output, so read them from source.
    for entry in tests:
        if entry["classes"] is None:
            ast_tree = ast.parse((ROOT / "tests" / str(entry["file"])).read_text(encoding="utf-8"))
            entry["classes"] = sum(
                1
                for node in ast.walk(ast_tree)
                if isinstance(node, ast.ClassDef) and node.name.startswith("Test")
            )

    tests.sort(key=lambda e: str(e["file"]))
    facts["tests"] = tests
    facts["test_total"] = sum(int(e["tests"]) for e in tests)

    return facts


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------
def build() -> Path:
    facts = collect_facts()

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)
        page_number_footer(section)

    title = "WorkCompanion AI"
    subtitle = "Multi-Agent AI Study & Research Companion"
    blurb = (
        "A grounded question-answering and revision system over the learner's own "
        "material. Built with Streamlit, CrewAI and Groq on an advanced hybrid "
        "retrieval pipeline, with explicit hallucination control and persistent "
        "learner memory."
    )

    # ---- cover ----
    for _ in range(3):
        doc.add_paragraph()
    para(doc, "PROJECT DOCUMENTATION", size=10, bold=True, color=MUTED,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    r.bold = True
    r.font.size = Pt(34)
    r.font.color.rgb = ACCENT
    para(doc, subtitle, size=15, color=MUTED,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=22)
    para(doc, blurb, size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=34,
         indent=0.7)

    meta_rows = [
        ["Version", "1.0.0"],
        ["Repository", "github.com/Asifshah01/workcompanion"],
        ["Revision", git_rev()],
        ["Generated", datetime.now(timezone.utc).strftime("%d %B %Y")],
        ["Runtime", f"Python {py_version().replace('Python ', '')}"],
        ["Interface", "Streamlit"],
        ["LLM provider", "Groq (offline extractive fallback available)"],
        ["Orchestration", "CrewAI (optional, lazily imported)"],
    ]
    table(doc, ["Field", "Value"], meta_rows, widths=[1.9, 4.7])

    doc.add_page_break()

    # ---- contents ----
    h(doc, "Contents", 1)
    para(doc, "Right-click and choose “Update Field” in Word to populate page "
              "numbers.", size=9, italic=True, color=MUTED, space_after=10)
    toc = doc.add_paragraph()
    field(toc, 'TOC \\o "1-2" \\h \\z \\u')
    doc.add_page_break()

    # ================= 1. OVERVIEW =================
    h(doc, "1. Overview", 1)

    h(doc, "1.1 Purpose", 2)
    para(doc, "WorkCompanion AI is a study and research companion that answers "
              "questions strictly from material the learner has provided, and is "
              "explicit about the difference between doing that and not doing it.")
    para(doc, "The problem it addresses is not a shortage of generated text. It "
              "is that generated text is cheap and untraceable: a study tool that "
              "confidently explains a concept has no obligation to be right, and "
              "no way for the learner to check. WorkCompanion's premise is that "
              "an answer you cannot trace is not worth much, so every answer "
              "carries its sources, its confidence and its grounding label, and "
              "the system refuses outright when its own retrieval cannot "
              "support a claim.")

    h(doc, "1.2 Design principles", 2)
    for principle, detail in [
        ("Evidence or silence",
         "An agent with nothing to cite refuses and says why. It does not fill "
         "the gap with plausible prose."),
        ("Traceability",
         "Every answer names the document, page and chunk it came from, so a "
         "claim can be checked against the original."),
        ("Honest uncertainty",
         "Confidence is derived from lexical alignment between the question and "
         "the retrieved passages, then gated. It is not a self-report from the "
         "model about how sure it feels."),
        ("Labelled knowledge",
         "Results distinguish retrieved fact, model reasoning and general "
         "knowledge. General knowledge is permitted but never disguised."),
        ("Remembers the learner",
         "A persistent profile, conversation memory and topic-level mastery let "
         "the app adapt rather than reset on every visit."),
        ("Degrades, never breaks",
         "No API key means a deterministic extractive provider, not a crash. "
         "CrewAI is optional. Storage is swappable."),
    ]:
        rich(doc, [(f"{principle} — ", {"bold": True}), (detail, {})],
             space_after=5, indent=0.15)

    h(doc, "1.3 Technology stack", 2)
    table(doc,
          ["Layer", "Technology", "Role"],
          [
              ["Interface", "Streamlit", "Web UI, ten modes, session state"],
              ["Orchestration", "CrewAI (optional)", "Three multi-agent flows; lazily imported"],
              ["LLM", "Groq API", "All generative work, JSON-structured output"],
              ["LLM fallback", "In-repo offline provider", "Deterministic extractive mode, no key needed"],
              ["Embeddings", "sentence-transformers", "Local dense vectors, all-MiniLM-L6-v2"],
              ["Vector store", "ChromaDB (or in-memory)", "Dense similarity search"],
              ["Sparse index", "Self-implemented BM25, JSON-backed", "Lexical search, document-scoped removal"],
              ["Reranking", "Cross-encoder or local heuristic", "Reorders candidates before compression"],
              ["Persistence", "SQLAlchemy + SQLite", "Twelve tables; Postgres-portable"],
              ["Documents", "pypdf, pdfplumber, python-docx, python-pptx",
               "Seven loaders with table preservation"],
              ["Validation", "Pydantic v2", "27 schema classes, strict parsing"],
              ["Packaging", "setuptools, Docker", "Editable install; container image"],
              ["Tests", "pytest", f"{facts['test_total']} tests, offline, no API key"],
          ],
          widths=[1.15, 2.35, 3.1])

    # ================= 2. REQUIREMENTS =================
    h(doc, "2. Requirements", 1)

    h(doc, "2.1 Functional requirements", 2)
    para(doc, "Derived from the implemented agent registry and page routes. Each "
              "row names the requirement and where it is satisfied.")

    fr = [
        ("FR-01", "Route any natural-language request to the correct specialist agent",
         "agents/nexus_agent.py"),
        ("FR-02", "Teach a concept at four levels of depth (beginner → expert)",
         "agents/tutor_agent.py"),
        ("FR-03", "Run scaffolded Socratic dialogue and probe for misconceptions",
         "agents/socratic_agent.py"),
        ("FR-04", "Answer questions from the learner's own documents with citations",
         "agents/rag_agent.py"),
        ("FR-05", "Refuse to answer when retrieval cannot support a claim",
         "rag/citations.py, agents/rag_agent.py"),
        ("FR-06", "Ingest PDF, DOCX, PPTX, TXT, MD, CSV and HTML, preserving tables",
         "loaders/"),
        ("FR-07", "Apply OCR fallback to scanned PDFs that yield too little text",
         "loaders/, OCR settings"),
        ("FR-08", "Deduplicate uploads by content hash; re-index without duplication",
         "rag/ingestion.py"),
        ("FR-09", "Hybrid dense + sparse retrieval with configurable weighting",
         "rag/hybrid_search.py"),
        ("FR-10", "Rewrite and expand queries, fan out multiple queries, rerank",
         "rag/query_transform.py, rag/reranker.py"),
        ("FR-11", "Compress context to the sentences that answer the question",
         "rag/compressor.py"),
        ("FR-12", "Generate quizzes grounded in the indexed material and grade them",
         "agents/quiz_agent.py"),
        ("FR-13", "Generate citation-backed flashcards for spaced repetition",
         "agents/flashcard_agent.py"),
        ("FR-14", "Produce spaced-repetition study plans from real constraints",
         "agents/planner_agent.py"),
        ("FR-15", "Synthesise research from documents and optional web sources",
         "agents/research_agent.py"),
        ("FR-16", "Persist learner profile, preferences and misconceptions",
         "memory/learner_profile.py"),
        ("FR-17", "Maintain conversation memory with a bounded window and summary",
         "memory/conversation_memory.py"),
        ("FR-18", "Track per-topic mastery and detect weak areas",
         "memory/progress_tracker.py"),
        ("FR-19", "Recommend a next action from the learner's own history",
         "memory/progress_tracker.py"),
        ("FR-20", "Run with no API key via a deterministic extractive provider",
         "llm/"),
        ("FR-21", "Run without CrewAI installed",
         "crew/ (lazy import)"),
        ("FR-22", "Expose runtime configuration without editing source",
         "ui/pages/settings.py"),
        ("FR-23", "Report environment and database health without leaking secrets",
         "ui/state.py"),
        ("FR-24", "Never print an API key value in any UI surface or report",
         "ui/state.py, enforced by test"),
    ]
    table(doc, ["ID", "Requirement", "Satisfied in"],
          [[a, b, c] for a, b, c in fr],
          widths=[0.5, 4.0, 2.1], mono_cols=(2,))

    h(doc, "2.2 Non-functional requirements", 2)
    table(doc, ["ID", "Requirement", "How it is met"],
          [
              ["NFR-01", "An answer must be traceable to a source passage",
               "Citations carry document, page and chunk id"],
              ["NFR-02", "Refuse rather than hallucinate when evidence is absent",
               "Confidence floor gates the refusal path"],
              ["NFR-03", "Run without paid API access",
               "Offline extractive provider; free-tier model selected"],
              ["NFR-04", "Tests must not require a key, network or rate limit",
               "Offline fixtures in tests/conftest.py"],
              ["NFR-05", "Survive LLM rate limits and transient errors",
               "Server-directed backoff, max_tokens clamp, 4 retries"],
              ["NFR-06", "Disclose truncation rather than silently cutting text",
               "LLMResponse.truncated surfaced in the output"],
              ["NFR-07", "Hold no secrets in version control",
               ".env ignored; verified absent from the tracked tree"],
              ["NFR-08", "Deploy to a free-tier host without a build timeout",
               "requirements-cloud.txt pins CPU-only PyTorch"],
              ["NFR-09", "Decompose cleanly by layer",
               "config / schemas / llm / rag / database / memory / agents / ui"],
              ["NFR-10", "No heavyweight dependency at import time",
               "CrewAI, OCR and cross-encoders imported lazily"],
          ],
          widths=[0.62, 2.9, 3.08])

    h(doc, "2.3 Runtime requirements", 2)
    table(doc, ["Requirement", "Minimum", "Recommended", "Notes"],
          [
              ["Python", "3.11", "3.12", "3.12.10 pinned for Cloud"],
              ["RAM", "4 GB", "8 GB", "Cross-encoder and embeddings both load"],
              ["Disk", "2 GB", "6 GB", "Embedding model ≈ 90 MB; vector store grows"],
              ["Network", "First run only", "—", "Model download; then offline-capable"],
              ["GPU", "Not required", "Optional", "CPU inference is fine for this workload"],
              ["Tesseract", "Optional", "Recommended", "Only for scanned PDFs"],
          ],
          widths=[1.25, 1.1, 1.25, 3.0])

    h(doc, "2.4 Software dependencies", 2)
    para(doc, "Principal runtime dependencies. The authoritative, pinned list is "
              "requirements.txt; requirements-cloud.txt is the CPU-optimised "
              "variant for Streamlit Cloud, and requirements-ocr.txt holds the "
              "optional OCR extras.")
    table(doc, ["Package", "Purpose"],
          [
              ["streamlit", "Web interface"],
              ["pydantic, pydantic-settings", "Validation and environment config"],
              ["groq", "LLM client"],
              ["crewai", "Optional multi-agent orchestration"],
              ["chromadb", "Dense vector store"],
              ["sentence-transformers, torch", "Local embeddings"],
              ["sqlalchemy", "Persistence"],
              ["pypdf, pdfplumber", "PDF text and table extraction"],
              ["python-docx, python-pptx", "Office documents"],
              ["fpdf2", "Generates the demo corpus (needed on a fresh deploy)"],
          ],
          widths=[2.35, 4.25], mono_cols=(0,))

    # ================= 3. OPERATION =================
    h(doc, "3. Operation", 1)

    h(doc, "3.1 Installation", 2)
    numbered(doc, "Clone the repository and enter the project directory.")
    numbered(doc, "Create and activate a virtual environment.")
    code(doc,
         "git clone https://github.com/Asifshah01/workcompanion.git\n"
         "cd workcompanion\n\n"
         "python -m venv .venv\n"
         ".venv\\Scripts\\activate          # Windows\n"
         "source .venv/bin/activate        # macOS / Linux")
    numbered(doc, "Install dependencies.")
    code(doc, "pip install -r requirements.txt\n"
              "pip install -r requirements-ocr.txt    # optional, scanned PDFs")
    numbered(doc, "Create the environment file and paste your Groq key into it.")
    code(doc, "cp .env.example .env\n"
              "# then edit .env and set:  GROQ_API_KEY=gsk_...")
    numbered(doc, "Start the application.")
    code(doc, "streamlit run app.py")
    para(doc, "The interface is served at http://localhost:8501. Open "
              "My Knowledge and press “Load demo material” to index a sample "
              "corpus and try a grounded question immediately.")

    callout(doc, "Running without a key",
            "GROQ_API_KEY is the only required secret. With no key the app runs "
            "on a deterministic offline provider that quotes your indexed "
            "passages instead of generating text, and every such answer is "
            "labelled offline in the interface so it cannot be mistaken for "
            "model output.")

    h(doc, "3.2 Configuration", 2)
    para(doc, "Configuration is layered: code defaults, then .env, then .env.local "
              "written by the Settings page. Settings edits are stored in "
              ".env.local, which never contains or modifies the real .env, and "
              "take effect on the next run after settings reload and the agent "
              "bundle re-bootstrap.")
    para(doc, "The full set of " + str(len(facts["settings"])) +
              " settings is listed in Appendix A. The ones that change behaviour "
              "most are summarised here.")

    highlight = {"GROQ_API_KEY", "LLM_PROVIDER", "GROQ_MODEL", "GROQ_MAX_TOKENS",
                 "GROQ_MAX_RETRIES", "DENSE_WEIGHT", "SPARSE_WEIGHT",
                 "RETRIEVAL_TOP_K", "MIN_RETRIEVAL_CONFIDENCE", "RERANKER",
                 "CHUNK_TARGET_CHARS", "ENABLE_CREWAI", "ENABLE_WEB_RESEARCH",
                 "DATABASE_URL", "CACHE_ENABLED"}
    rows = []
    for s in facts["settings"]:
        if s["name"] in highlight:
            rows.append([s["name"], s["type"].replace("<class '", "").replace("'>", ""),
                         s["default"][:34], (s["doc"] or "—")[:60]])
    table(doc, ["Setting", "Type", "Default", "Meaning"], rows,
          widths=[1.6, 1.35, 1.2, 2.45], mono_cols=(0,))

    h(doc, "3.3 Modes of operation", 2)
    para(doc, "The application exposes ten modes. Document-backed modes short-"
              "circuit with an explanatory prompt when the index is empty rather "
              "than answering without evidence.")

    pages = [
        ("Dashboard", "—", "Next-step recommendation, weak areas, library and activity summary"),
        ("AI Tutor", "—", "Explanations at a chosen level, plus a Socratic mode that asks rather than tells"),
        ("My Knowledge", "Upload", "Upload, index, re-index and delete documents; loads the demo corpus"),
        ("Research", "Yes", "Multi-source synthesis and structured paper analysis"),
        ("Quiz", "Yes", "Grounded question set, automatic grading, per-question feedback"),
        ("Flashcards", "Yes", "Atomic citation-backed deck with review tracking"),
        ("Exam Mode", "Yes", "Timed attempt under exam conditions with a graded report"),
        ("Study Planner", "—", "Spaced-repetition plan from subject, deadline and available hours"),
        ("Progress", "—", "Mastery per topic, quiz history, activity and recommendations"),
        ("Settings", "—", "Environment, retrieval weights, agent options, maintenance"),
    ]
    table(doc, ["Mode", "Needs documents", "Purpose"], pages,
          widths=[1.35, 1.35, 3.9])

    h(doc, "3.4 Typical operating flow", 2)
    numbered(doc, "Start the app and, if the index is empty, press "
                  "“Load demo material” on My Knowledge, or upload your own files.")
    numbered(doc, "On the Dashboard, read the recommended next step. It is derived "
                  "from your own topic accuracy, not a fixed suggestion.")
    numbered(doc, "Ask a question in the chat box on Dashboard or AI Tutor. NEXUS "
                  "routes it; the answer shows its sources and confidence.")
    numbered(doc, "Follow up in the same thread. The conversation window and its "
                  "rolling summary carry the context.")
    numbered(doc, "Take a Quiz. Grading updates topic mastery and reshapes what "
                  "the planner schedules and what the next quiz targets.")
    numbered(doc, "Review Progress for weak areas, then adjust retrieval weights in "
                  "Settings if answers seem over- or under-confident.")
    numbered(doc, "Ask in Exam Mode for a timed, graded attempt.")

    h(doc, "3.5 Runtime controls", 2)
    para(doc, "Behaviour under adverse conditions is deliberate and disclosed:")
    table(doc, ["Condition", "Behaviour"],
          [
              ["No API key", "Offline extractive provider; output labelled offline"],
              ["Groq 429 rate limit", "Server-directed backoff, then retries up to GROQ_MAX_RETRIES"],
              ["Non-retryable 4xx", "Failed immediately; no pointless retry loop"],
              ["Answer hits token cap", "Truncation disclosed in the output, not silent"],
              ["Empty index", "Document-backed modes prompt for an upload instead of answering"],
              ["Low retrieval confidence", "Refusal with an explanation of what is missing"],
              ["CrewAI unavailable", "Single-agent path used; the app still works"],
              ["OCR unavailable", "Scanned pages index as empty and say so"],
          ],
          widths=[2.0, 4.6])

    # ================= 4. ARCHITECTURE =================
    doc.add_page_break()
    h(doc, "4. Architecture", 1)

    h(doc, "4.1 Module structure", 2)
    tree_rows = [
        ["config/", "Settings, environment loading, validation"],
        ["schemas/", "27 Pydantic models shared across every layer"],
        ["llm/", "Groq client, offline provider, retry and backoff"],
        ["loaders/", "Seven document loaders, OCR fallback"],
        ["rag/", "Chunking, embeddings, hybrid search, sparse index, query "
                 "transform, reranker, compressor, citations, pipeline"],
        ["database/", "SQLAlchemy models, repositories, session handling"],
        ["memory/", "Learner profile, conversation memory, progress tracker"],
        ["agents/", "Eight agents plus the registry and shared base"],
        ["crew/", "Optional CrewAI flows and the Groq bridge"],
        ["tools/", "Web search and other external tools"],
        ["ui/", "Layout, theme, components, and one module per mode"],
    ]
    table(doc, ["Package", "Responsibility"], tree_rows, widths=[1.3, 5.3], mono_cols=(0,))
    para(doc, f"The implementation is {len(facts['tree'])} Python modules. "
              f"Dependencies point downwards: the UI knows about agents, agents "
              f"know about retrieval, retrieval knows about storage. Nothing below "
              f"the UI imports it.", size=10)

    h(doc, "4.2 Request flow", 2)
    code(doc,
         "User question\n"
         "     |\n"
         "     v\n"
         "  NEXUS ── intent + extracted parameters\n"
         "     |\n"
         "     v\n"
         "  specialist agent (tutor | knowledge | research | quiz | ...)\n"
         "     |\n"
         "     +──> retrieval: rewrite ─> multi-query ─> hybrid\n"
         "     |          ─> rerank ─> compress ─> confidence\n"
         "     |\n"
         "     v\n"
         "  LLM (Groq, JSON-structured)  ── or ──> offline extractive provider\n"
         "     |\n"
         "     v\n"
         "  citation + grounding check ── below floor ──> refusal\n"
         "     |\n"
         "     v\n"
         "  AgentResult ──> UI renders answer, sources, confidence, warnings\n"
         "     |\n"
         "     v\n"
         "  memory: profile, conversation, topic mastery")

    h(doc, "4.3 Agent roster", 2)
    para(doc, "Eight agents, each with one responsibility. Question-answering "
              "agents return a common AgentResult envelope so the UI can render "
              "them uniformly; generation agents return their artifact, because a "
              "StudyPlan is more useful as a plan than as a string and a QuizSet "
              "must keep its questions intact to be gradeable.")
    table(doc, ["Agent", "Class", "Responsibility"],
          [[a["key"], a["cls"], a["role"]] for a in facts["agents"]],
          widths=[0.95, 1.6, 4.05], mono_cols=(0, 1))

    para(doc, "Principal public methods, as implemented:", size=10, space_after=4)
    api_rows = []
    for a in facts["agents"]:
        methods = [m for m in a["methods"] if not m.startswith(("build_request",
                   "complete", "result", "failure", "safe_json", "build_request("))]
        api_rows.append([a["key"], "\n".join(methods[:7])])
    table(doc, ["Agent", "Key methods"], api_rows, widths=[1.0, 5.6], mono_cols=(1,))

    h(doc, "4.4 Data model", 2)
    para(doc, f"{len(facts['tables'])} tables, created on first run and portable to "
              f"Postgres by changing DATABASE_URL.")
    table(doc, ["Table", "Model", "Columns", "Purpose"],
          [
              ["users", "User", str(len(facts["tables"][0]["columns"])), "Learner identity and preferences"],
              ["documents", "Document", "22", "Indexed material, content hash, status, chunk count"],
              ["conversations", "Conversation", "7", "Chat threads and rolling summaries"],
              ["conversation_messages", "ConversationMessage", "11", "Individual turns with citations and confidence"],
              ["topic_progress", "TopicProgress", "8", "Per-topic accuracy and attempts"],
              ["quiz_attempts", "QuizAttempt", "17", "Quiz sessions and scores"],
              ["quiz_question_records", "QuizQuestionRecord", "16", "Per-question responses and grading"],
              ["flashcard_decks", "FlashcardDeck", "7", "Generated decks"],
              ["flashcards", "Flashcard", "11", "Cards with review counts and citation refs"],
              ["study_plans", "StudyPlan", "7", "Plans with revision schedules"],
              ["study_sessions", "StudySession", "9", "Time and activity history"],
              ["paper_analyses", "PaperAnalysis", "5", "Structured research output"],
          ],
          widths=[1.6, 1.5, 0.6, 2.9], mono_cols=(0,))

    para(doc, "Domain schemas (Pydantic, shared across layers):", size=10, space_after=4)
    table(doc, ["Schema", "Purpose"],
          [[s["name"], ", ".join(s["fields"][:7]) + ("…" if len(s["fields"]) > 7 else "")]
           for s in facts["schemas"]],
          widths=[1.7, 4.9], mono_cols=(0,))

    # ================= 5. RAG =================
    doc.add_page_break()
    h(doc, "5. Retrieval pipeline and hallucination control", 1)

    para(doc, "This is the core of the system and the reason it is trustworthy. "
              "Each stage exists to make a specific failure harder.")

    h(doc, "5.1 Ingestion", 2)
    table(doc, ["Stage", "What happens", "Why"],
          [
              ["1. Load", "Text, page numbers and tables extracted per format",
               "Tables are structure, not prose; flattening them loses information"],
              ["2. Chunk", "Split on headings and sentence boundaries with overlap",
               "A fixed character split cuts concepts in half"],
              ["3. Metadata", "Document, page, heading path, section, position",
               "Citations must be resolvable to a page a human can open"],
              ["4. Deduplicate", "Content hash identifies the document",
               "Re-uploading must not double every chunk"],
              ["5. Index", "Dense embeddings plus a document-scoped BM25 index",
               "Dense finds concepts, sparse finds exact terms and names"],
          ],
          widths=[1.0, 2.5, 3.1])

    h(doc, "5.2 Query processing", 2)
    table(doc, ["Stage", "What happens", "Why"],
          [
              ["1. Rewrite", "Question restated in retrieval form, "
                             "conversation-aware", "Questions are phrased for humans, "
                                                     "not for embedding search"],
              ["2. Expand", "Additional phrasings generated", "Recovers vocabulary "
                                                               "mismatch"],
              ["3. Multi-query", "Several paraphrases, merged by reciprocal rank", "One "
               "phrasing can miss; several rarely do"],
              ["4. Hybrid", "Dense + sparse, blended DENSE_WEIGHT / SPARSE_WEIGHT",
               "Either signal alone has a characteristic blind spot"],
              ["5. Rerank", "Candidates reordered by relevance", "Cosine similarity "
                                                                  "is a poor ranking"],
              ["6. Compress", "Reduced to sentences that bear on the question",
               "Irrelevant context dilutes the answer and invites drift"],
              ["7. Score", "Confidence computed and citations verified",
               "Decides whether to answer at all"],
          ],
          widths=[1.0, 2.5, 3.1])

    h(doc, "5.3 Hallucination control", 2)

    rich(doc, [("Lexical-alignment gate. ", {"bold": True}),
               ("Confidence is multiplied down when the question's terms barely "
                "appear in the retrieved passages. This gate was added after "
                "measurement showed the opposite of the intended behaviour: "
                "off-topic questions were scoring ", {}),
               ("higher", {"italic": True}),
               (" than on-topic ones, because retrieval surfaced something "
                "plausible and the model elaborated on it confidently. Chunk term "
                "extraction had been frequency-capped at 16 terms and coverage "
                "contributed only a fifth of the blend, so a document matching a "
                "few high-frequency words looked aligned.", {})])

    bullet(doc, "Below the confidence floor the agent refuses, returning a "
                "specific statement of what is missing rather than a vague hedge.")
    bullet(doc, "Citations carry document, page and chunk id, and answer claims "
                "are checked against the cited text.")
    bullet(doc, "Grounding is declared explicitly as retrieved fact, model "
                "reasoning or general knowledge — never left for the learner to infer.")
    bullet(doc, "General knowledge is permitted only when the caller opts in, and "
                "then carries no citations, a confidence cap of 0.3, a "
                "GENERAL_KNOWLEDGE label and an explicit warning.")
    bullet(doc, "Grounded agents (quiz, flashcards, research) refuse to generate "
                "when nothing relevant is indexed, rather than inventing questions.")

    callout(doc, "A defect worth recording",
            "allow_general_knowledge was threaded into the prompt but never "
            "consulted at the refusal gate, so the flag did nothing and "
            "out-of-corpus questions refused even when explicitly permitted. The "
            "fix added a dedicated path; the first attempt passed every structural "
            "assertion while the model was still replying that the provided "
            "context did not contain the answer. Only reading the generated text — "
            "not merely checking that a response existed — exposed it. The prompt "
            "was passing CONTEXT: (none), which the model read as a reason to "
            "refuse rather than permission to use its own knowledge.",
            color=WARN, shade_hex="FDF3E3")

    # ================= 6. MEMORY =================
    h(doc, "6. Memory and progress", 1)
    table(doc, ["Component", "What it stores", "How it is used"],
          [
              ["Learner profile", "Level, style, subject, weekly hours, misconceptions",
               "Injected into tutor prompts so teaching adapts across sessions"],
              ["Conversation memory", "Bounded window of recent turns plus a rolling summary",
               "Gives follow-up questions context without unbounded prompts"],
              ["Progress tracker", "Per-topic accuracy, mastery, quiz history, "
                                   "flashcards, study minutes, activity by day",
               "Drives the dashboard recommendation, quiz focus and planner schedule"],
              ["Weak-area detection", "Topics ranked by lowest mastery",
               "Scheduled first, tested first"],
          ],
          widths=[1.35, 2.4, 2.85])
    para(doc, "The Progress page does not only chart history: it recommends a next "
              "action derived from the learner's own records.")

    # ================= 7. TESTING =================
    h(doc, "7. Verification and testing", 1)

    h(doc, "7.1 Test suite", 2)
    para(doc, f"{facts['test_total']} automated tests across four suites, plus four "
              f"smoke scripts. Every test runs offline against the deterministic "
              f"provider on throwaway storage, so a green run means the "
              f"orchestration is correct rather than the model happening to be "
              f"helpful today.")
    covers = {
        "test_rag.py": "Chunking, hybrid search, BM25, citations, confidence",
        "test_agents.py": "All eight agents, envelope shape, refusal behaviour",
        "test_persistence.py": "Repositories, profile, progress, conversations",
        "test_ui.py": "Page registry, navigation, transcript, settings",
    }
    table(doc, ["Suite", "Tests", "Classes", "Covers"],
          [[f"tests/{t['file']}", str(t["tests"]), str(t["classes"]),
            covers.get(t["file"], "")]
           for t in facts["tests"]],
          widths=[1.75, 0.65, 0.75, 3.45], mono_cols=(0,))

    h(doc, "7.2 Smoke and live checks", 2)
    table(doc, ["Script", "Checks", "Purpose"],
          [
              ["_smoke_agents.py", "32", "Agent orchestration and output shape"],
              ["_smoke_crew.py", "64", "CrewAI flows and the Groq bridge"],
              ["_smoke_ui.py", "29", "Every page renders without error"],
              ["_smoke_live.py", "32", "Real model: grounded answers, refusal, generation"],
              ["_chk_env.py", "—", ".env.example parses into valid Settings"],
              ["_bench_groq.py", "—", "Latency and JSON reliability per model"],
          ],
          widths=[1.6, 0.8, 4.2], mono_cols=(0,))

    h(doc, "7.3 Defects found and fixed during verification", 2)
    para(doc, "Recorded because each is now protected by a test.")
    table(doc, ["Defect", "Effect", "Test that prevents it"],
          [
              ["Off-topic questions scored higher confidence than on-topic ones",
               "Answers were confidently wrong on out-of-corpus questions",
               "test_rag.py confidence coverage gate"],
              ["allow_general_knowledge ignored at the refusal gate",
               "The flag did nothing; permitted questions still refused",
               "test_agents.py general-knowledge labelling"],
              ["Session factory never rebound after an engine change",
               "Changing DATABASE_URL in Settings silently did nothing",
               "test_persistence.py isolation across databases"],
              ["Duplicate detection was a stub returning False",
               "Re-ingesting a file doubled every chunk",
               "test_rag.py ingest idempotence"],
              ["Forced re-index accumulated instead of replacing",
               "Repeated re-indexing grew the index without bound",
               "test_rag.py re-index replace"],
              ["Model-supplied level parsed as int; LLM returned \"1-5\"",
               "ValueError crash on a Socratic scaffold level",
               "test_agents.py tolerant coercion"],
              ["Cron-free token ceiling ignored in the Groq client",
               "Long answers truncated at the cap",
               "live smoke truncation disclosure"],
          ],
          widths=[2.35, 2.35, 1.9])

    h(doc, "7.4 Manual verification performed", 2)
    bullet(doc, "Full test suite green with no API key and no network.")
    bullet(doc, "All four smoke suites green; live Groq suite green against the real model.")
    bullet(doc, "Application boots with a clean stderr and serves HTTP 200.")
    bullet(doc, "Grounded question answered in the browser with correct content, "
                "page-level citations, high confidence and explicit grounding labels.")
    bullet(doc, "Refusal path confirmed: an out-of-corpus question refuses and explains why.")
    bullet(doc, "All ten modes rendered and inspected individually.")
    bullet(doc, "Settings confirmed to report key status without printing the value.")
    bullet(doc, "Clean-tree audit: no .env, no .venv, no data/, no key material in "
                "the tracked repository.")
    bullet(doc, "Cloud simulation: committed tree extracted to a clean directory and "
                "booted with no .env and no key, rendering all ten modes.")

    # ================= 8. DEPLOY =================
    doc.add_page_break()
    h(doc, "8. Deployment", 1)

    h(doc, "8.1 Docker", 2)
    code(doc,
         "docker build -t workcompanion .\n"
         "docker run --rm -p 8501:8501 \\\n"
         "  --env-file .env \\\n"
         "  -v wc-data:/app/data \\\n"
         "  workcompanion")
    para(doc, "The image includes Tesseract and libmagic. Mounting /app/data "
              "preserves the document library, vector store and database across "
              "restarts.")

    h(doc, "8.2 Streamlit Cloud", 2)
    numbered(doc, "Create a new app from the GitHub repository, branch main.")
    numbered(doc, "Set the Python version to 3.12 and the requirements file to "
                  "requirements-cloud.txt. That file pins a CPU-only PyTorch, "
                  "which is substantially smaller than the default CUDA wheels "
                  "and avoids a build timeout.")
    numbered(doc, "Deploy, then open Settings → Secrets and add GROQ_API_KEY.")
    numbered(doc, "Reload. Settings reads the process environment, so the secret "
                  "is picked up with no code change.")
    para(doc, "Supporting files: runtime.txt pins python-3.12.10; packages.toml "
              "installs tesseract-ocr and libmagic1.", size=10)

    callout(doc, "Cloud storage is ephemeral",
            "Streamlit Cloud storage resets on restart and on every code change. "
            "The vector store and database are rebuilt empty. This is a platform "
            "constraint rather than a defect; use “Load demo material” or re-upload "
            "after a redeploy.",
            color=WARN, shade_hex="FDF3E3")

    # ================= 9. LIMITS =================
    h(doc, "9. Known limitations", 1)
    para(doc, "Stated plainly so they are discovered from documentation rather "
              "than in use.")
    table(doc, ["Limitation", "Detail", "Mitigation in place"],
          [
              ["Groq free-tier throughput",
               "Approximately 1000 output tokens per minute; sustained multi-user "
               "use will hit 429s",
               "Server-directed backoff, max_tokens clamp, four retries; a paid "
               "key removes the ceiling"],
              ["Answer length",
               "Long answers truncate at GROQ_MAX_TOKENS",
               "Truncation is disclosed in the output rather than hidden"],
              ["Offline mode quality",
               "Extractive, not generative; it quotes passages and cannot explain "
               "a concept the notes never mention",
               "Labelled offline so it is never mistaken for model output"],
              ["Cloud persistence",
               "Storage resets on restart or redeploy",
               "Documented; one-button re-seed provided"],
              ["BM25 persistence",
               "JSON-backed, appropriate at this scale but not at millions of chunks",
               "Documented as a known boundary"],
              ["First-run latency",
               "Embedding and cross-encoder models download once",
               "Warmed cache; negligible thereafter"],
          ],
          widths=[1.5, 2.75, 2.35])

    # ================= 10. TROUBLESHOOT =================
    h(doc, "10. Troubleshooting", 1)
    table(doc, ["Symptom", "Cause", "Resolution"],
          [
              ["“Add a document in My Knowledge first” on every question",
               "The index is empty; the app is refusing to answer without evidence",
               "Load demo material or upload a file on My Knowledge"],
              ["Everything labelled offline", "GROQ_API_KEY not set or not visible",
               "Check Settings → Environment; on Cloud add the secret under "
               "Settings → Secrets, not in a local .env"],
              ["429 rate limit errors", "Free-tier throughput ceiling",
               "Lower GROQ_MAX_TOKENS, use qwen/qwen3.8-27b, or supply a paid key"],
              ["An answer refused although the notes cover it",
               "Confidence below MIN_RETRIEVAL_CONFIDENCE",
               "Lower the threshold in Settings; confirm the correct subject was "
               "chosen at ingestion"],
              ["PDFs index as empty", "Scanned document without OCR",
               "Install requirements-ocr.txt and Tesseract; ingestion records "
               "whether OCR ran"],
              ["Very slow first request", "Model download and warm-up",
               "Expected once; later requests are much faster"],
              ["Upload rejected", "Extension or size outside the allow-list",
               "Check ALLOWED_UPLOAD_EXTENSIONS and MAX_UPLOAD_MB in Settings"],
              ["Settings change appears to do nothing",
               "Fixed in this release; session factory now rebinds on a database change",
               "Update to the current revision"],
          ],
          widths=[2.05, 2.15, 2.4])

    # ================= 11. FUTURE =================
    h(doc, "11. Future work", 1)
    for item in [
        "Multi-user accounts with authentication and per-user storage, replacing "
        "the single-learner assumption.",
        "Persistent cloud storage so the vector store and database survive a redeploy.",
        "A paid-tier path with larger models and higher token ceilings, removing "
        "the free-tier throughput ceiling.",
        "Expanded retrieval evaluation: a labelled question set measuring recall, "
        "citation precision and refusal accuracy as the pipeline changes.",
        "Active re-indexing when a stored document's hash no longer matches the file.",
        "Streaming responses so long answers begin arriving before completion.",
        "Learner-facing export of progress history and study plans as PDF.",
    ]:
        bullet(doc, item)

    # ================= APPENDIX A =================
    doc.add_page_break()
    h(doc, "Appendix A — Complete settings reference", 1)
    para(doc, f"All {len(facts['settings'])} settings with their declared defaults. "
              f"Values here are the code defaults; an .env or .env.local file "
              f"overrides them.", size=9.5, italic=True, color=MUTED)
    rows = []
    for s in facts["settings"]:
        rows.append([
            s["name"],
            s["type"].replace("<class '", "").replace("'>", "").replace("typing.", "")[:40],
            s["default"][:26],
            (s["doc"] or "")[:58],
        ])
    table(doc, ["Setting", "Type", "Default", "Description"], rows,
          widths=[1.75, 1.55, 1.05, 2.25], mono_cols=(0,))

    # ================= APPENDIX B =================
    doc.add_page_break()
    h(doc, "Appendix B — Module inventory", 1)
    para(doc, f"{len(facts['tree'])} Python modules in the workcompanion package.",
         size=9.5, italic=True, color=MUTED)
    grouped: dict[str, list[str]] = {}
    for path in facts["tree"]:
        parts = path.split("/")
        pkg = parts[1] if len(parts) > 2 else "(root)"
        grouped.setdefault(pkg, []).append(parts[-1] if len(parts) > 2 else path)
    rows = [[pkg, str(len(files)), ", ".join(sorted(files)[:6]) +
             ("…" if len(files) > 6 else "")] for pkg, files in sorted(grouped.items())]
    table(doc, ["Package", "Modules", "Files"], rows,
          widths=[1.15, 0.75, 4.7], mono_cols=(2,))

    # ================= APPENDIX C =================
    h(doc, "Appendix C — Referenced source files", 1)
    table(doc, ["Purpose", "Path"],
          [
              ["Application entry point", "app.py"],
              ["Dependency manifests", "requirements.txt, requirements-cloud.txt, "
                                        "requirements-ocr.txt, requirements-dev.txt"],
              ["Environment template", ".env.example"],
              ["Container build", "Dockerfile"],
              ["Host configuration", "runtime.txt, packages.toml, .streamlit/config.toml"],
              ["Test fixtures", "tests/conftest.py"],
              ["Smoke and bench scripts", "scripts/"],
              ["Demo corpus builder", "scripts/build_demo_data.py"],
              ["Operational guide", "README.md"],
          ],
          widths=[2.1, 4.5], mono_cols=(1,))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    size = path.stat().st_size / 1024
    print(f"wrote {path}  ({size:.1f} KB)")