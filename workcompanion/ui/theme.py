"""Visual language for WorkCompanion AI.

Two things live here: the CSS injected once per Streamlit run, and a tiny
palette module so every page colours the same concept the same way (confidence,
grounding labels, agent names).  Colour is never the *only* signal - every
badge also carries a word, so the UI stays readable for colour-blind learners.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from workcompanion.schemas.common import ConfidenceLevel, GroundingLabel

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
#: Single accent ramp used for gradients, borders and focus rings.
ACCENT = "#6C5CE7"
ACCENT_SOFT = "#A29BFE"
ACCENT_DARK = "#4834D4"

#: Semantic colours. Paired with text labels everywhere they are used.
SEMANTIC = {
    "high": "#00B894",
    "medium": "#FDCB6E",
    "low": "#E17055",
    "none": "#636E72",
}

GROUNDING_COLOURS = {
    GroundingLabel.RETRIEVED_FACT: "#00B894",
    GroundingLabel.MODEL_REASONING: "#0984E3",
    GroundingLabel.GENERAL_KNOWLEDGE: "#E17055",
}

GROUNDING_ICONS = {
    GroundingLabel.RETRIEVED_FACT: "📄",
    GroundingLabel.MODEL_REASONING: "🧠",
    GroundingLabel.GENERAL_KNOWLEDGE: "🌐",
}

CONFIDENCE_ICONS = {
    ConfidenceLevel.HIGH: "🟢",
    ConfidenceLevel.MEDIUM: "🟡",
    ConfidenceLevel.LOW: "🟠",
}

#: Emoji per agent, used in the sidebar roster and the trace panel.
AGENT_ICONS = {
    "nexus": "🧭",
    "knowledge": "📚",
    "tutor": "🎓",
    "socratic": "🤔",
    "research": "🔬",
    "quiz": "📝",
    "flashcards": "🗂️",
    "planner": "🗓️",
    "crew": "👥",
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = f"""
<style>
  :root {{
    --wc-accent: {ACCENT};
    --wc-accent-soft: {ACCENT_SOFT};
    --wc-accent-dark: {ACCENT_DARK};
    --wc-high: {SEMANTIC['high']};
    --wc-medium: {SEMANTIC['medium']};
    --wc-low: {SEMANTIC['low']};
    --wc-none: {SEMANTIC['none']};
  }}

  /* Sidebar ------------------------------------------------------------ */
  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #14121f 0%, #1b1830 100%);
    border-right: 1px solid rgba(255,255,255,.06);
  }}
  [data-testid="stSidebar"] * {{ color: #ece9f7; }}
  [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.10); }}

  /* Brand -------------------------------------------------------------- */
  .wc-brand {{
    display: flex; align-items: center; gap: .7rem;
    padding: .2rem 0 .6rem 0;
  }}
  .wc-brand-mark {{
    width: 2.6rem; height: 2.6rem; border-radius: .8rem;
    background: linear-gradient(135deg, {ACCENT}, {ACCENT_SOFT});
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; flex: 0 0 auto;
    box-shadow: 0 6px 18px rgba(108,92,231,.35);
  }}
  .wc-brand-title {{ font-weight: 700; font-size: 1.06rem; line-height: 1.15; }}
  .wc-brand-sub {{ font-size: .72rem; opacity: .65; letter-spacing: .04em; }}

  /* Hero --------------------------------------------------------------- */
  .wc-hero {{
    padding: 1.35rem 1.5rem; border-radius: 1.1rem; margin-bottom: 1rem;
    background: linear-gradient(120deg, {ACCENT_DARK} 0%, {ACCENT} 55%, #8E7CF0 100%);
    color: #fff; box-shadow: 0 12px 32px rgba(72,52,212,.28);
  }}
  .wc-hero h1 {{ margin: 0 0 .3rem 0; font-size: 1.65rem; }}
  .wc-hero p {{ margin: 0; opacity: .92; font-size: .95rem; }}

  /* Stat tiles --------------------------------------------------------- */
  .wc-stats {{ display: flex; flex-wrap: wrap; gap: .7rem; margin: .2rem 0 1rem 0; }}
  .wc-stat {{
    flex: 1 1 8.5rem; background: rgba(108,92,231,.07);
    border: 1px solid rgba(108,92,231,.18); border-left: 3px solid {ACCENT};
    border-radius: .75rem; padding: .6rem .8rem;
  }}
  .wc-stat-value {{ font-size: 1.35rem; font-weight: 700; line-height: 1.1; }}
  .wc-stat-label {{
    font-size: .7rem; text-transform: uppercase; letter-spacing: .07em; opacity: .6;
  }}

  /* Badges ------------------------------------------------------------- */
  .wc-badges {{ display: flex; flex-wrap: wrap; gap: .35rem; margin: .15rem 0 .6rem 0; }}
  .wc-badge {{
    display: inline-flex; align-items: center; gap: .3rem;
    padding: .12rem .55rem; border-radius: 999px;
    font-size: .72rem; font-weight: 600; border: 1px solid transparent;
  }}
  .wc-badge-high   {{ background: rgba(0,184,148,.14);  color: {SEMANTIC['high']};   border-color: rgba(0,184,148,.35); }}
  .wc-badge-medium {{ background: rgba(253,203,110,.18); color: #B8891B;              border-color: rgba(253,203,110,.45); }}
  .wc-badge-low    {{ background: rgba(225,112,85,.14);  color: {SEMANTIC['low']};    border-color: rgba(225,112,85,.35); }}
  .wc-badge-none   {{ background: rgba(99,110,114,.14);  color: {SEMANTIC['none']};   border-color: rgba(99,110,114,.3); }}
  .wc-badge-neutral{{ background: rgba(108,92,231,.12);  color: {ACCENT_DARK};      border-color: rgba(108,92,231,.3); }}

  /* Confidence bar ----------------------------------------------------- */
  .wc-meter {{
    height: 6px; border-radius: 999px; background: rgba(127,127,127,.18);
    overflow: hidden; margin: .3rem 0 .1rem 0;
  }}
  .wc-meter > span {{ display: block; height: 100%; border-radius: 999px; }}
  .wc-meter-high > span   {{ background: {SEMANTIC['high']}; }}
  .wc-meter-medium > span {{ background: {SEMANTIC['medium']}; }}
  .wc-meter-low > span    {{ background: {SEMANTIC['low']}; }}
  .wc-meter-none > span   {{ background: {SEMANTIC['none']}; }}

  /* Source cards ------------------------------------------------------- */
  .wc-source {{
    border-left: 3px solid {ACCENT_SOFT}; background: rgba(162,155,254,.08);
    border-radius: .5rem; padding: .5rem .75rem; margin: .35rem 0; font-size: .82rem;
  }}
  .wc-source-title {{ font-weight: 600; }}
  .wc-source-meta  {{ opacity: .65; font-size: .75rem; }}
  .wc-quote {{ opacity: .8; font-style: italic; margin-top: .25rem; }}

  /* Agent cards -------------------------------------------------------- */
  .wc-agent {{
    display: flex; align-items: center; gap: .6rem; padding: .55rem .7rem;
    border-radius: .7rem; background: rgba(108,92,231,.06);
    border: 1px solid rgba(108,92,231,.15); margin-bottom: .35rem;
  }}
  .wc-agent-icon {{ font-size: 1.35rem; }}
  .wc-agent-name {{ font-weight: 650; font-size: .9rem; }}
  .wc-agent-desc {{ font-size: .74rem; opacity: .62; }}

  /* Workflow / trace --------------------------------------------------- */
  .wc-trace-row {{ display: flex; align-items: center; gap: .5rem; padding: .3rem 0; }}
  .wc-trace-dot {{
    width: .55rem; height: .55rem; border-radius: 50%; background: {ACCENT}; flex: 0 0 auto;
  }}
  .wc-trace-key {{ font-weight: 600; font-size: .82rem; }}
  .wc-trace-note {{ font-size: .74rem; opacity: .6; }}

  /* Warnings ----------------------------------------------------------- */
  .wc-warning {{
    border-left: 3px solid {SEMANTIC['low']}; background: rgba(225,112,85,.10);
    padding: .45rem .7rem; border-radius: .45rem; margin: .3rem 0; font-size: .82rem;
  }}

  /* Flashcards --------------------------------------------------------- */
  .wc-card {{
    border: 1px solid rgba(108,92,231,.22); border-radius: .8rem;
    padding: .8rem 1rem; margin-bottom: .5rem; background: rgba(255,255,255,.5);
  }}
  .wc-card-front {{ font-weight: 650; }}
  .wc-card-back {{ margin-top: .4rem; opacity: .9; }}

  /* Misc --------------------------------------------------------------- */
  .wc-pill-row {{ display: flex; gap: .4rem; flex-wrap: wrap; margin: .3rem 0 .6rem 0; }}
  .wc-hint {{ font-size: .78rem; opacity: .6; }}
  .wc-kv {{ font-size: .8rem; }}
  .wc-kv b {{ display: inline-block; min-width: 8.5rem; opacity: .65; font-weight: 600; }}

  /* Streamlit tweaks --------------------------------------------------- */
  div.stButton > button {{
    border-radius: .6rem; border: 1px solid rgba(108,92,231,.35);
    font-weight: 600;
  }}
  div.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {ACCENT}, {ACCENT_DARK});
    border: none; color: #fff;
  }}
  section[data-testid="stExpander"] {{
    border: 1px solid rgba(108,92,231,.16); border-radius: .7rem; background: rgba(255,255,255,.35);
  }}
  .stTabs [data-baseweb="tab-list"] {{ gap: .2rem; }}
</style>
"""


def inject_css() -> None:
    """Attach the stylesheet to the running Streamlit app (idempotent)."""
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def level_key(level: Any) -> str:
    """Normalise a confidence level to one of the four CSS modifier keys."""
    value = getattr(level, "value", level)
    text = str(value or "none").lower()
    return text if text in SEMANTIC else "none"


def confidence_icon(level: Any) -> str:
    """Emoji for a confidence level, tolerating unknown values."""
    return CONFIDENCE_ICONS.get(level, "⚪")


def brand(subtitle: str | None = None) -> None:
    """Sidebar brand block."""
    st.markdown(
        f"""
        <div class="wc-brand">
          <div class="wc-brand-mark">🎓</div>
          <div>
            <div class="wc-brand-title">WorkCompanion<span style="opacity:.6"> AI</span></div>
            <div class="wc-brand-sub">{subtitle or "MULTI-AGENT STUDY COPILOT"}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str = "") -> None:
    """Page header with the accent gradient."""
    st.markdown(
        f'<div class="wc-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def stat_row(stats: list[tuple[str, Any]]) -> None:
    """A row of label/value tiles."""
    tiles = "".join(
        f'<div class="wc-stat"><div class="wc-stat-value">{value}</div>'
        f'<div class="wc-stat-label">{label}</div></div>'
        for label, value in stats
    )
    st.markdown(f'<div class="wc-stats">{tiles}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "neutral", icon: str = "") -> None:
    """One coloured pill."""
    modifier = kind if kind in SEMANTIC or kind == "neutral" else "neutral"
    st.markdown(
        f'<span class="wc-badge wc-badge-{modifier}">{icon} {text}</span>',
        unsafe_allow_html=True,
    )


def badge_row(badges: list[tuple[str, str, str]]) -> None:
    """Several pills in a row. Each item is ``(text, kind, icon)``."""
    if not badges:
        return
    inner = "".join(
        f'<span class="wc-badge wc-badge-{(k if k in SEMANTIC else "neutral")}">{i} {t}</span>'
        for t, k, i in badges
    )
    st.markdown(f'<div class="wc-badges">{inner}</div>', unsafe_allow_html=True)


def confidence_meter(score: float, level: Any, label: str = "confidence") -> None:
    """Bar + pill showing how well grounded an answer is."""
    key = level_key(level)
    percent = max(0.0, min(1.0, float(score))) * 100
    name = getattr(level, "value", level) or "none"
    st.markdown(
        f"""
        <div class="wc-meter wc-meter-{key}"><span style="width:{percent:.1f}%"></span></div>
        <div class="wc-kv"><b>{label}</b> {confidence_icon(level)}
        {name} · {float(score):.0%}</div>
        """,
        unsafe_allow_html=True,
    )


def grounding_row(labels: Any) -> None:
    """Show which claims came from the documents, the model, or general knowledge."""
    items = list(labels or [])
    if not items:
        return
    counts: dict[Any, int] = {}
    for label in items:
        counts[label] = counts.get(label, 0) + 1
    badges = [
        (
            f"{getattr(label, 'value', label).replace('_', ' ')} ×{count}",
            "neutral",
            GROUNDING_ICONS.get(label, "•"),
        )
        for label, count in counts.items()
    ]
    badge_row(badges)


def agent_card(name: str, description: str = "") -> None:
    """A single roster row."""
    st.markdown(
        f"""
        <div class="wc-agent">
          <div class="wc-agent-icon">{AGENT_ICONS.get(name, "🤖")}</div>
          <div>
            <div class="wc-agent-name">{name.replace('_', ' ').title()}</div>
            <div class="wc-agent-desc">{description}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def trace(stages: list[str], notes: dict[str, str] | None = None) -> None:
    """The pipeline stages a request went through."""
    notes = notes or {}
    rows = "".join(
        f'<div class="wc-trace-row"><div class="wc-trace-dot"></div>'
        f'<div><div class="wc-trace-key">{stage}</div>'
        f'<div class="wc-trace-note">{notes.get(stage, "")}</div></div></div>'
        for stage in stages
    )
    st.markdown(f'<div>{rows}</div>', unsafe_allow_html=True)


def warning(text: str) -> None:
    """Inline non-blocking warning."""
    st.markdown(f'<div class="wc-warning">⚠️ {text}</div>', unsafe_allow_html=True)


def hint(text: str) -> None:
    st.markdown(f'<div class="wc-hint">{text}</div>', unsafe_allow_html=True)


def kv(pairs: list[tuple[str, Any]]) -> None:
    """Label/value list (redacts anything already masked)."""
    body = "".join(f"<div><b>{label}</b>{value}</div>" for label, value in pairs)
    st.markdown(f'<div class="wc-kv">{body}</div>', unsafe_allow_html=True)


__all__ = [
    "ACCENT",
    "AGENT_ICONS",
    "CONFIDENCE_ICONS",
    "GROUNDING_COLOURS",
    "GROUNDING_ICONS",
    "SEMANTIC",
    "agent_card",
    "badge",
    "badge_row",
    "brand",
    "confidence_icon",
    "confidence_meter",
    "grounding_row",
    "hero",
    "hint",
    "inject_css",
    "kv",
    "level_key",
    "stat_row",
    "trace",
    "warning",
]