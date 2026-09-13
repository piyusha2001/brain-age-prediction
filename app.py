import sys
from pathlib import Path
import json
from html import escape
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from textwrap import dedent

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


from model_service import (
    analyze_patient,
    load_model_artifacts,
)

from pdf_parser import (
    parse_patient_pdf,
)

from evidence_pipeline import (
    build_evidence_from_model_result,
)

from claude_client import (
    build_patient_explanation,
)

from elevenlabs_client import (
    build_factor_narration,
    build_report_narration,
    build_welcome_narration,
    generate_friday_audio,
)


def attach_analysis_values(report, analysis_result):
    """Carry calculated display values into the explanatory UI report."""

    report_factors = report.get("factors", [])
    analysis_factors = analysis_result.get("factors", [])

    if len(report_factors) != len(analysis_factors):
        raise ValueError(
            "The explanation did not preserve all calculated factors."
        )

    for report_factor, analysis_factor in zip(
        report_factors,
        analysis_factors,
    ):
        report_factor["shap_impact"] = analysis_factor["shap_impact"]
        report_factor["z_score"] = analysis_factor["z_score"]
        report_factor["patient_level"] = analysis_factor["patient_level"]

    return report


@st.cache_data
def load_cached_demo_report():

    demo_path = (
        PROJECT_ROOT
        / "outputs"
        / "patient_2561_explanation.json"
    )

    if not demo_path.exists():
        raise FileNotFoundError(
            f"Demo JSON not found: {demo_path}"
        )

    with open(
        demo_path,
        "r",
        encoding="utf-8"
    ) as file:

        report = json.load(file)

    evidence_path = PROJECT_ROOT / "outputs" / "patient_2561_amass_evidence.json"
    if evidence_path.exists():
        with open(evidence_path, "r", encoding="utf-8") as file:
            evidence = json.load(file)
        report = attach_analysis_values(report, evidence)

    return report


@st.cache_data(show_spinner=False)
def load_friday_audio(narration):
    return generate_friday_audio(narration)



# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(
    page_title="Synapse AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root { --ink:#edf4f3; --muted:#8e9e9f; --teal:#51d7d0; --line:rgba(125,164,163,.18); --panel:rgba(13,27,35,.76); }
    .stApp { color:var(--ink); background:radial-gradient(circle at 82% 6%,rgba(37,125,121,.15),transparent 28rem),radial-gradient(circle at 12% 38%,rgba(39,70,110,.14),transparent 30rem),#071119; font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
    header[data-testid="stHeader"] { background:transparent; }
    #MainMenu, footer { visibility:hidden; }
    div[data-testid="stDialog"] { background:rgba(2,8,13,.72)!important; backdrop-filter:blur(10px); }
    div[data-testid="stDialog"]>div { display:flex!important; align-items:center!important; justify-content:center!important; width:100%!important; max-width:none!important; }
    div[data-testid="stDialog"] [role="dialog"] { width:min(780px,92vw)!important; max-width:780px!important; max-height:90vh!important; border:1px solid rgba(81,215,208,.2)!important; border-radius:24px!important; background:#0b151e!important; color:#e8f1f1!important; box-shadow:0 30px 90px rgba(0,0,0,.5)!important; }
    div[data-testid="stDialog"] [role="dialog"]>div { background:transparent!important; }
    div[data-testid="stDialog"] h2 { color:#eef7f6!important; font-size:1.45rem!important; letter-spacing:-.025em; }
    div[data-testid="stDialog"] label,div[data-testid="stDialog"] p,div[data-testid="stDialog"] span { color:#9aabad!important; }
    div[data-testid="stDialog"] button[aria-label*="Close"] { color:#9fb2b5!important; background:transparent!important; border:0!important; outline:0!important; box-shadow:none!important; }
    div[data-testid="stDialog"] button[aria-label*="Close"]:hover { color:#edf7f6!important; background:rgba(255,255,255,.06)!important; }
    div[data-testid="stDialog"] button[kind="secondary"] { background:#101d27!important; border:1px solid rgba(145,174,178,.25)!important; }
    div[data-testid="stDialog"] button[kind="secondary"] p { color:#c8d4d5!important; }
    div[data-testid="stDialog"] button[kind="primary"] p { color:#061316!important; }
    div[data-testid="stDialog"] [data-testid="stCheckbox"] { padding:.15rem 0; }
    div[data-testid="stDialog"] [data-testid="stCheckbox"] label span { color:#c1cdcf!important; line-height:1.45; }
    .policy-intro { color:#899b9e; font-size:.8rem; line-height:1.55; margin:-.15rem 0 .85rem; }
    .policy-intro strong { display:inline-block; margin-top:.15rem; color:#51d7d0; font-weight:600; }
    .policy-scroll { height:min(48vh,440px); overflow-y:auto; padding:.1rem .9rem .4rem 0; scrollbar-width:thin; scrollbar-color:#31515a transparent; }
    .policy-warning { padding:.85rem 1rem; margin:.25rem 0 .75rem; border:1px solid rgba(139,161,165,.16); border-radius:12px; background:rgba(255,255,255,.025); color:#8fa0a3; font-size:.72rem; line-height:1.55; }
    .policy-warning strong { color:#c6d2d3; }
    .policy-card { padding:.9rem .15rem; margin:0; border:0; border-bottom:1px solid rgba(139,161,165,.12); border-radius:0; background:transparent; }
    .policy-card h4 { margin:0 0 .42rem; color:#dce7e7; font-size:.82rem; font-weight:600; }
    .policy-card p,.policy-card li,.policy-card td,.policy-card th { color:#8fa0a3!important; font-size:.69rem; line-height:1.55; }
    .policy-card ul { margin:.35rem 0 .2rem; padding-left:1.2rem; }
    .policy-card table { width:100%; border-collapse:collapse; }
    .policy-card th,.policy-card td { padding:.35rem; border-bottom:1px solid rgba(139,161,165,.12); text-align:left; vertical-align:top; }
    .policy-tag { display:inline-block; padding:.25rem .5rem; border:1px solid rgba(81,215,208,.24); border-radius:999px; color:#75d9d4; background:rgba(81,215,208,.06); font-size:.58rem; letter-spacing:.08em; text-transform:uppercase; }
    .intro-transition { position:fixed; inset:0; z-index:999999; display:grid; place-items:center; overflow:hidden; pointer-events:none; background:#071119; animation:intro-overlay 4.1s cubic-bezier(.68,0,.2,1) forwards; }
    .intro-brand { position:absolute; z-index:2; color:#effafa; font-size:clamp(3rem,8vw,8rem); font-weight:700; letter-spacing:-.07em; animation:intro-word 3.5s ease forwards; }
    .intro-brand span { color:var(--teal); }
    .intro-brain { width:min(66vw,760px); opacity:.2; filter:drop-shadow(0 0 45px rgba(81,215,208,.38)); animation:intro-brain 4s cubic-bezier(.7,0,.2,1) forwards; }
    .intro-brain path { fill:rgba(31,70,88,.85); stroke:#51d7d0; stroke-width:3; }
    .intro-play .hero-copy { animation:copy-arrive 1.1s 2.9s ease both; }
    .intro-play .brain-visual { animation:hero-brain-arrive 1.4s 2.65s cubic-bezier(.2,.8,.2,1) both; }
    @keyframes intro-word { 0%{opacity:0;transform:scale(.84);filter:blur(12px)} 25%,62%{opacity:1;transform:scale(1);filter:blur(0)} 100%{opacity:0;transform:translateX(-18vw) scale(.72);filter:blur(5px)} }
    @keyframes intro-brain { 0%{opacity:0;transform:scale(1.5) rotate(-5deg)} 25%,58%{opacity:.42;transform:scale(1) rotate(0)} 85%{opacity:.5;transform:translateX(34vw) scale(.42)} 100%{opacity:0;transform:translateX(38vw) scale(.34)} }
    @keyframes intro-overlay { 0%,78%{opacity:1;visibility:visible} 100%{opacity:0;visibility:hidden} }
    @keyframes copy-arrive { from{opacity:0;transform:translateX(-35px)} to{opacity:1;transform:translateX(0)} }
    @keyframes hero-brain-arrive { from{opacity:0;transform:translateX(-30vw) scale(1.75)} to{opacity:1;transform:translateX(0) scale(1)} }

    .block-container {
        max-width:none; width:100%; padding:1.25rem clamp(1.25rem,4vw,5rem) 6rem;
    }

    h1,h2,h3 { color:var(--ink)!important; font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important; }
    h1,h2 { font-weight:600!important; }
    p,label,[data-testid="stCaptionContainer"] { color:var(--muted); }
    .topbar { display:flex; align-items:center; justify-content:space-between; padding:.4rem 0 1.1rem; border-bottom:1px solid var(--line); }
    .wordmark { font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:1.15rem; font-weight:600; letter-spacing:-.03em; color:var(--ink); }
    .wordmark>span:first-child { color:var(--teal); }
    .nav-note { font-size:.66rem; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); }
    .pill { margin-left:.65rem; padding:.3rem .62rem; border:1px solid rgba(81,215,208,.35); border-radius:999px; color:var(--teal); font-family:'Inter',sans-serif; font-size:.6rem; letter-spacing:.14em; text-transform:uppercase; }

    .hero {
        min-height:calc(100vh - 4rem); padding:clamp(3rem,7vh,6rem) 0; margin-bottom:1rem;
        display:grid; grid-template-columns:minmax(0,1.05fr) minmax(420px,.95fr); gap:clamp(2rem,5vw,6rem); align-items:center;
    }

    .hero h1 {
        font-size:clamp(2.8rem,4.4vw,5rem); letter-spacing:-.05em; line-height:1.06; margin:0 0 2rem; max-width:850px;
        transform-origin:left center;
    }
    .hero h1:hover { color:#dffefd!important; animation:title-bounce 2.4s ease-in-out infinite; }

    .hero p {
        font-size:clamp(1rem,1.25vw,1.25rem); line-height:1.8; max-width:700px; margin:0;
    }

    .eyebrow { color:var(--teal); font-size:.66rem; letter-spacing:.22em; text-transform:uppercase; margin-bottom:1.2rem; }
    .hero em { color:var(--teal); font-weight:400; }
    .hero-copy { position:relative; z-index:2; }
    .hero-actions { display:flex; gap:.8rem; flex-wrap:wrap; margin:2.5rem 0 2.7rem; }
    .hero-button { display:inline-flex; align-items:center; justify-content:center; min-height:54px; padding:0 1.8rem; border-radius:999px; border:1px solid var(--line); color:var(--ink)!important; text-decoration:none!important; font-weight:600; }
    .hero-button.primary { color:#061316!important; background:var(--teal); border-color:var(--teal); box-shadow:0 12px 34px rgba(81,215,208,.16); }
    .hero-button:hover { border-color:rgba(81,215,208,.7); }
    .brain-visual { position:relative; width:min(100%,700px); justify-self:center; filter:drop-shadow(0 20px 60px rgba(24,106,117,.16)); }
    .brain-visual svg { width:100%; height:auto; overflow:visible; }
    .brain-outline { fill:url(#brainFill); stroke:#348a98; stroke-width:3.2; stroke-linecap:round; stroke-linejoin:round; }
    .brain-detail { fill:none; stroke:#367583; stroke-width:2.2; stroke-linecap:round; stroke-linejoin:round; opacity:.9; }
    .brain-circuit { fill:none; stroke:#386d79; stroke-width:1.6; opacity:.8; }
    .brain-node { fill:#72d9dc; filter:drop-shadow(0 0 5px rgba(114,217,220,.8)); animation:node-glow 2.8s ease-in-out infinite; }
    .brain-node.dim { opacity:.32; animation-delay:1.2s; }
    .node-hotspot { cursor:crosshair; }
    .node-hotspot .brain-node { transition:r .2s ease,filter .2s ease; }
    .node-hotspot:hover .brain-node { r:11px; filter:drop-shadow(0 0 10px rgba(114,217,220,1)); }
    .node-popup { display:inline-block; opacity:0; transform:translateY(7px) scale(.94); transform-origin:left bottom; padding:7px 10px; border:1px solid rgba(81,215,208,.42); border-radius:9px; background:rgba(6,17,25,.96); color:#eaffff; font-family:Inter,sans-serif; font-size:10px; font-weight:600; line-height:1.25; white-space:normal; box-shadow:0 10px 25px rgba(0,0,0,.35); pointer-events:none; transition:opacity .18s ease,transform .18s ease; }
    .node-hotspot:hover .node-popup { opacity:1; transform:translateY(0) scale(1); }
    .scan-ring { fill:none; stroke:#2d7180; stroke-width:1.2; opacity:.3; transform-origin:center; animation:ring-breathe 5s ease-in-out infinite; }
    .target-ring { fill:none; stroke:#52d3dc; stroke-width:3; transform-origin:center; animation:target-pulse 2.6s ease-in-out infinite; }
    .brain-stem { fill:none; stroke:#367f8e; stroke-width:10; stroke-linecap:round; }
    .brain-tag { position:absolute; left:50%; bottom:1%; transform:translateX(-50%); width:max-content; max-width:92%; padding:.8rem 1.5rem; border:1px solid rgba(160,174,182,.24); border-radius:999px; background:rgba(10,18,26,.9); color:#a9b1bc; font-size:.7rem; letter-spacing:.2em; text-transform:uppercase; }
    @keyframes node-glow { 0%,100%{opacity:.45} 50%{opacity:1} }
    @keyframes ring-breathe { 0%,100%{transform:scale(.98);opacity:.22} 50%{transform:scale(1.025);opacity:.45} }
    @keyframes target-pulse { 0%,100%{transform:scale(.85);opacity:.7} 50%{transform:scale(1.12);opacity:1} }
    @keyframes title-bounce { 0%,100%{transform:translateY(0)} 35%{transform:translateY(-7px)} 60%{transform:translateY(2px)} 80%{transform:translateY(-2px)} }
    .intro { max-width:720px; margin:0 auto 3rem; text-align:center; }
    #how-it-works, #try-neurodecel { scroll-margin-top:2rem; }
    .intro h2 { font-size:2.35rem; margin-bottom:.8rem; }
    .intro p { line-height:1.75; }
    .feature-card { min-height:185px; padding:1.6rem; border:1px solid var(--line); border-radius:18px; background:linear-gradient(145deg,rgba(16,33,42,.86),rgba(8,19,27,.72)); }
    .feature-card,[data-testid="stMetric"] { transition:transform .3s ease,border-color .3s ease,box-shadow .3s ease; }
    .feature-card:hover,[data-testid="stMetric"]:hover { transform:translateY(-4px); border-color:rgba(81,215,208,.38); box-shadow:0 18px 42px rgba(0,0,0,.18); }
    .feature-number { color:var(--teal); font-size:.65rem; letter-spacing:.18em; }
    .feature-card h3 { font-size:1rem; margin:2rem 0 .7rem; }
    .feature-card p { font-size:.86rem; line-height:1.6; }
    .section-kicker { color:var(--teal); font-size:.65rem; letter-spacing:.18em; text-transform:uppercase; margin-top:4rem; }
    .section-heading { font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:2.5rem; font-weight:600; letter-spacing:-.04em; color:var(--ink); margin:.5rem 0 1.5rem; }
    .report-shell { margin-top:3rem; animation:soft-arrive .65s ease both; }
    .report-kicker { color:var(--teal); font-size:.68rem; letter-spacing:.2em; text-transform:uppercase; }
    .report-title { font-size:clamp(2.2rem,4vw,4rem); font-weight:600; letter-spacing:-.045em; margin:.5rem 0 1.6rem; }
    .age-track { position:relative; height:12px; margin:2.2rem 1rem 1.2rem; border-radius:999px; background:linear-gradient(90deg,#254a55,#51d7d0,#254a55); }
    .age-marker { position:absolute; top:50%; width:18px; height:18px; border:4px solid #071119; border-radius:50%; background:#e8ffff; transform:translate(-50%,-50%); box-shadow:0 0 0 3px var(--teal),0 0 20px rgba(81,215,208,.55); }
    .age-labels { display:flex; justify-content:space-between; color:var(--muted); font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; }
    .insight-panel { padding:1.6rem; border:1px solid var(--line); border-radius:18px; background:var(--panel); }
    .contribution-row { margin:0 0 1.35rem; }
    .contribution-head { display:flex; justify-content:space-between; gap:1rem; margin-bottom:.5rem; font-size:.88rem; }
    .contribution-head span:last-child { color:var(--muted); white-space:nowrap; }
    .contribution-track { position:relative; height:10px; border-radius:999px; background:#18232d; overflow:hidden; }
    .contribution-mid { position:absolute; left:50%; top:0; bottom:0; width:1px; background:rgba(255,255,255,.2); }
    .contribution-bar { position:absolute; top:0; bottom:0; border-radius:999px; }
    .contribution-bar.younger { right:50%; background:linear-gradient(90deg,#3b7c83,#70dadd); }
    .contribution-bar.older { left:50%; background:linear-gradient(90deg,#9a724e,#f0ac70); }
    .axis-labels { display:flex; justify-content:space-between; margin-top:1rem; color:var(--muted); font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; }
    .biomarker-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; margin:1.4rem 0 2.5rem; }
    .biomarker-card { padding:1.35rem; border:1px solid var(--line); border-radius:18px; background:linear-gradient(145deg,rgba(19,32,42,.9),rgba(8,18,26,.78)); transition:.3s ease; }
    .biomarker-card:hover { transform:translateY(-3px); border-color:rgba(81,215,208,.36); }
    .biomarker-top { display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; }
    .biomarker-name { color:var(--ink); font-weight:600; }
    .biomarker-icon { display:grid; place-items:center; flex:none; width:36px; height:36px; border:1px solid rgba(81,215,208,.28); border-radius:11px; background:rgba(81,215,208,.07); color:var(--teal); font-size:1.1rem; }
    .biomarker-badge { flex:none; padding:.32rem .65rem; border:1px solid var(--line); border-radius:999px; color:#a8b3bc; font-size:.6rem; letter-spacing:.1em; text-transform:uppercase; }
    .biomarker-badge.notable { border-color:rgba(81,215,208,.48); color:var(--teal); }
    .biomarker-desc { margin:.8rem 0!important; font-size:.84rem; line-height:1.55; }
    .biomarker-meta { color:var(--muted); font-size:.68rem; letter-spacing:.09em; text-transform:uppercase; }
    .reference-mini { position:relative; height:6px; margin:1rem 0 .45rem; border-radius:999px; background:linear-gradient(90deg,rgba(239,170,108,.6),#263843 28%,#376873 50%,#263843 72%,rgba(239,170,108,.6)); }
    .reference-mini span { position:absolute; top:50%; width:13px; height:13px; transform:translate(-50%,-50%); border:3px solid #0d1b24; border-radius:50%; background:#dfffff; box-shadow:0 0 0 2px var(--teal),0 0 10px rgba(81,215,208,.5); }
    .reference-labels { display:flex; justify-content:space-between; color:#71838b; font-size:.55rem; letter-spacing:.08em; text-transform:uppercase; }
    .biomarker-card details { margin-top:.9rem; padding-top:.75rem; border-top:1px solid var(--line); }
    .biomarker-card summary { color:var(--teal); cursor:pointer; font-size:.7rem; font-weight:600; list-style:none; }
    .biomarker-card summary::-webkit-details-marker { display:none; }
    .biomarker-card details p { margin:.65rem 0 0!important; font-size:.76rem; line-height:1.55; }
    .factor-heading { display:flex; gap:.6rem; flex-wrap:wrap; align-items:center; margin-bottom:1rem; }
    .factor-pill { padding:.3rem .62rem; border:1px solid var(--line); border-radius:999px; color:var(--teal); font-size:.63rem; letter-spacing:.08em; text-transform:uppercase; }
    .analysis-carousel { display:flex; gap:1.15rem; overflow-x:auto; padding:.5rem .15rem 1.5rem; margin:1rem 0 2.5rem; scroll-snap-type:x mandatory; scroll-behavior:smooth; scrollbar-color:rgba(81,215,208,.45) transparent; scrollbar-width:thin; }
    .analysis-carousel::-webkit-scrollbar { height:6px; }
    .analysis-carousel::-webkit-scrollbar-thumb { background:rgba(81,215,208,.4); border-radius:999px; }
    .analysis-card { position:relative; flex:0 0 clamp(315px,31vw,445px); min-height:410px; padding:1.5rem; overflow:hidden; scroll-snap-align:start; border:1px solid var(--line); border-radius:22px; background:radial-gradient(circle at 85% 8%,rgba(81,215,208,.1),transparent 9rem),linear-gradient(145deg,rgba(20,35,45,.96),rgba(8,18,27,.94)); transition:transform .35s ease,border-color .35s ease; }
    .analysis-card:hover { transform:translateY(-5px); border-color:rgba(81,215,208,.45); }
    .analysis-index { color:var(--teal); font-size:.65rem; letter-spacing:.18em; text-transform:uppercase; }
    .analysis-card h3 { min-height:3rem; margin:.7rem 0 .25rem; font-size:1.05rem; line-height:1.45; }
    .analysis-subtitle { color:var(--muted); font-size:.78rem; }
    .mini-dashboard { display:grid; grid-template-columns:132px 1fr; gap:1rem; align-items:center; margin:1.5rem 0; }
    .z-gauge { position:relative; width:128px; height:128px; }
    .z-gauge svg { width:100%; transform:rotate(-90deg); }
    .gauge-bg { fill:none; stroke:#1b2b35; stroke-width:9; }
    .gauge-progress { fill:none; stroke:var(--teal); stroke-width:9; stroke-linecap:round; filter:drop-shadow(0 0 5px rgba(81,215,208,.5)); }
    .gauge-center { position:absolute; inset:0; display:grid; place-content:center; text-align:center; }
    .gauge-number { color:var(--ink); font-size:1.35rem; font-weight:700; }
    .gauge-label { color:var(--muted); font-size:.56rem; letter-spacing:.12em; text-transform:uppercase; }
    .mini-stat { padding:.75rem 0; border-bottom:1px solid var(--line); }
    .mini-stat:last-child { border:0; }
    .mini-stat-label { color:var(--muted); font-size:.58rem; letter-spacing:.12em; text-transform:uppercase; }
    .mini-stat-value { margin-top:.25rem; color:var(--ink); font-size:.85rem; font-weight:600; }
    .microbar { height:7px; margin:.55rem 0 .15rem; border-radius:999px; background:#1a2730; overflow:hidden; }
    .microbar span { display:block; height:100%; border-radius:inherit; background:linear-gradient(90deg,#397782,#73dade); animation:bar-grow .9s ease both; transform-origin:left; }
    .microbar.older span { background:linear-gradient(90deg,#9b724c,#efaa6d); }
    .evidence-dots { display:flex; gap:.32rem; margin-top:.5rem; }
    .evidence-dots i { width:7px; height:7px; border-radius:50%; background:#24343e; }
    .evidence-dots i.on { background:var(--teal); box-shadow:0 0 7px rgba(81,215,208,.45); }
    .analysis-takeaway { margin:0!important; padding-top:1rem; border-top:1px solid var(--line); font-size:.8rem; line-height:1.55; }
    .carousel-hint { display:flex; align-items:center; gap:.55rem; color:var(--muted); font-size:.72rem; }
    .carousel-hint span { color:var(--teal); animation:hint-slide 1.8s ease-in-out infinite; }
    .voice-caption { color:var(--muted); font-size:.7rem; }
    [data-testid="stAudio"] { display:none!important; }
    [class*="st-key-friday_trigger_"] { display:none!important; }
    @keyframes soft-arrive { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
    @keyframes bar-grow { from{transform:scaleX(0)} to{transform:scaleX(1)} }
    @keyframes hint-slide { 0%,100%{transform:translateX(0)} 50%{transform:translateX(5px)} }

    .section-title {
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .age-card {
        padding: 1.5rem;
        border: 1px solid rgba(120,120,120,0.2);
        border-radius: 18px;
        text-align: center;
    }

    .big-age {
        font-size: 4rem;
        font-weight: 700;
        line-height: 1;
    }

    .subtle {
        opacity: 0.65;
        font-size: 0.92rem;
    }

    .younger {
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        background: rgba(50, 180, 120, 0.12);
        display: inline-block;
    }

    .older {
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        background: rgba(230, 140, 60, 0.12);
        display: inline-block;
    }

    .disclaimer {
        padding:1.25rem 1.4rem; border:1px solid var(--line); border-radius:14px; background:var(--panel);
        margin-top: 2rem;
        font-size: 0.9rem;
    }

    [data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); padding:1.2rem; border-radius:14px; }
    [data-testid="stMetricValue"] { color:var(--ink); font-family:'Inter',sans-serif; }
    [data-testid="stExpander"] { background:var(--panel); border:1px solid var(--line); border-radius:14px; overflow:hidden; }
    [data-testid="stFileUploaderDropzone"] { background:rgba(14,29,38,.82); border:1px dashed rgba(81,215,208,.38); border-radius:14px; }
    [data-baseweb="tab-list"] { gap:.5rem; border-bottom:1px solid var(--line); }
    button[data-baseweb="tab"] { background:transparent; }
    button[kind="primary"] { background:var(--teal)!important; color:#061316!important; border:0!important; border-radius:999px!important; font-weight:600!important; }
    button[kind="primary"] p, button[kind="primary"] span { color:#061316!important; opacity:1!important; }
    button[kind="secondary"] { border-color:var(--line)!important; border-radius:999px!important; }
    hr { border-color:var(--line)!important; }
    @media(max-width:1100px) { .hero{grid-template-columns:1fr}.brain-visual{max-width:620px;grid-row:1} }
    @media(max-width:700px) { .block-container{padding-left:1rem;padding-right:1rem}.hero{min-height:auto;padding:3rem 0}.hero h1{font-size:2.8rem}.brain-visual{max-width:430px}.brain-tag{font-size:.54rem;padding:.65rem 1rem}.nav-note{display:none}.feature-card{min-height:auto;margin-bottom:.75rem}.biomarker-grid{grid-template-columns:1fr}.contribution-head{font-size:.78rem}.analysis-card{flex-basis:88vw}.mini-dashboard{grid-template-columns:112px 1fr}.z-gauge{width:108px;height:108px} }
    @media(prefers-reduced-motion:reduce) { .brain-node,.scan-ring,.target-ring,.hero h1,.microbar span,.carousel-hint span,.intro-transition,.intro-brand,.intro-brain,.intro-play .hero-copy,.intro-play .brain-visual{animation:none}.intro-transition{display:none} }

    </style>
    """,
    unsafe_allow_html=True,
)


POLICY_VERSION = "v0.1-prototype"

POLICY_HTML = """
<div class="policy-intro">SynapseAI — Brain-Age Insight Platform<br><strong>Terms &amp; Conditions and Privacy Policy</strong></div>
<div class="policy-scroll">
  <div class="policy-warning"><strong>Prototype notice.</strong> This document was drafted for a hackathon prototype, covers GDPR and the EU AI Act, and has not been reviewed by a lawyer. It requires full legal review before any real personal or health data is processed or the platform is used outside demo/testing.</div>
  <div class="policy-card"><h4>Contents</h4><p>1 Who we are · 2 What SynapseAI is · 3 Data we collect · 4 Legal basis · 5 How we use data · 6 Processors and transfers · 7 Retention · 8 Security · 9 Your rights · 10 EU AI Act · 11 Children · 12 Medical and liability terms · 13 Changes · 14 Contact</p></div>
  <div class="policy-card"><h4>1. Who we are</h4><p>SynapseAI is the data controller under GDPR (Regulation (EU) 2016/679). Founders: Piyusha Patil, Shakeel J, Soundharya Y, and Haripriya Sampath. Contact: jshakeelw@gmail.com, ppiyusha2001@gmail.com, haripriya.sampath5@gmail.com, Soundharyaa18@gmail.com.</p></div>
  <div class="policy-card"><h4>2. What SynapseAI is — and is not</h4><p>SynapseAI is a research and educational prototype. It estimates “brain age” from structural MRI-derived measurements, explains which measurements shaped the estimate, and connects those explanations to published science.</p><span class="policy-tag">Not a medical device</span><p>It does not diagnose, predict health outcomes, life expectancy, or disease risk; it does not replace medical advice; and its outputs must never be the sole basis for a medical decision.</p></div>
  <div class="policy-card"><h4>3. Data we collect</h4><table><tr><th>Category</th><th>Examples</th><th>Special category?</th></tr><tr><td>Structural brain measurements</td><td>MRI-derived values</td><td>Yes — health data, Art. 9</td></tr><tr><td>Demographic data</td><td>Age; sex if provided</td><td>May relate to health analysis</td></tr><tr><td>Uploaded documents</td><td>PDFs containing scan-derived data</td><td>Yes, when health-related</td></tr><tr><td>Usage/technical data</td><td>Basic service activity</td><td>No</td></tr><tr><td>Feedback</td><td>Comments about the prototype</td><td>Usually no</td></tr></table><p>We do not knowingly collect the name or direct identifiers of the person whose scan is analyzed.</p></div>
  <div class="policy-card"><h4>4. Legal basis for processing</h4><p>Special-category health data is processed with explicit informed consent under GDPR Art. 9(2)(a), given by accepting this policy. General usage data may be processed under legitimate interests, Art. 6(1)(f). Consent can be withdrawn at any time without affecting processing already lawfully completed.</p></div>
  <div class="policy-card"><h4>5. How we use your data</h4><ul><li>Generate a brain-age estimate.</li><li>Identify top contributing measurements with SHAP.</li><li>Retrieve relevant scientific evidence.</li><li>Create a plain-language summary.</li><li>Improve the service with de-identified or aggregated data.</li></ul><p>We do not sell data or use it for unrelated advertising or profiling.</p></div>
  <div class="policy-card"><h4>6. Third-party processors and international transfers</h4><table><tr><th>Processor</th><th>Purpose</th><th>Data shared</th></tr><tr><td>Anthropic (Claude API)</td><td>Explanation and summarization</td><td>Relevant result and evidence context</td></tr><tr><td>Amass</td><td>Literature retrieval and evidence grading</td><td>Searchable biomarker context</td></tr><tr><td>ElevenLabs</td><td>Optional text-to-speech</td><td>Text selected for narration</td></tr></table><p>Transfers outside the EEA use GDPR Chapter V safeguards, such as Standard Contractual Clauses, or an adequacy decision where applicable.</p></div>
  <div class="policy-card"><h4>7. Data retention</h4><p>Data is kept only as long as needed for its stated purpose. Hackathon/demo data is retained for the event and evaluation period, then deleted, unless longer retention is requested for features such as tracking changes across scans or is required by law.</p></div>
  <div class="policy-card"><h4>8. Security measures</h4><ul><li>Encryption in transit using HTTPS/TLS.</li><li>Restricted access to raw uploads.</li><li>API keys stored separately from personal and health data.</li><li>Regular review of processor security.</li></ul><p>No system guarantees absolute security. Breach notification follows GDPR Arts. 33–34, including authority notification within 72 hours where required and user notification without undue delay for high-risk breaches.</p></div>
  <div class="policy-card"><h4>9. Your GDPR rights</h4><p>Access (Art. 15), rectification (16), erasure (17), restriction (18), portability (20), objection (21), withdrawal of consent (7(3)), and lodging a complaint with a supervisory authority (77).</p></div>
  <div class="policy-card"><h4>10. EU AI Act transparency notice</h4><ul><li>Outputs are disclosed as AI-generated.</li><li>The prototype is positioned as an educational and decision-support tool, not a medical device.</li><li>It is not intended to fall within the Act’s high-risk medical-AI category; legal counsel must reassess classification before clinical deployment.</li><li>AI claims carry a confidence level: High, Moderate, Low, or Nothing reliable found.</li><li>Human oversight is central: outputs can inform a conversation with a doctor, not replace one.</li></ul></div>
  <div class="policy-card"><h4>11. Children’s data</h4><p>The service is not directed to users under 16, or the applicable national digital-consent age, without parent or guardian consent. Children’s health data is not knowingly processed without it.</p></div>
  <div class="policy-card"><h4>12. Medical disclaimer and limitation of liability</h4><p>The prototype is provided “as is,” without a warranty of accuracy or fitness for purpose. It must not form the basis of medical, diagnostic, or treatment decisions. Liability is disclaimed to the fullest extent permitted by law, without limiting rights or liabilities that cannot legally be excluded under EU consumer or data-protection law.</p></div>
  <div class="policy-card"><h4>13. Changes to this policy</h4><p>This policy may be updated. Material changes to health-data handling require renewed explicit consent.</p></div>
  <div class="policy-card"><h4>14. Contact and complaints</h4><p>Contact the founders at jshakeelw@gmail.com, ppiyusha2001@gmail.com, haripriya.sampath5@gmail.com, or Soundharyaa18@gmail.com. You may lodge a complaint with your local EU supervisory authority at any time.</p><p>Last updated: 12/09/2026 · Shakeel J · Soundharya Y · Haripriya Sampath · Piyusha Patil</p></div>
</div>
"""


@st.dialog("Before you continue", width="large")
def show_policy_dialog():
    st.markdown(POLICY_HTML, unsafe_allow_html=True)
    acknowledged = st.checkbox(
        "I have read and agree to the Terms & Conditions and Privacy "
        "Policy above, including the processing of health-related data "
        "described in Section 4.",
        key="policy_acknowledged",
    )
    cancel_col, accept_col = st.columns([1, 1.7])
    with cancel_col:
        if st.button("Cancel", use_container_width=True):
            st.session_state["policy_declined"] = True
            st.rerun()
    with accept_col:
        if st.button(
            "Accept and continue",
            type="primary",
            use_container_width=True,
            disabled=not acknowledged,
        ):
            st.session_state["policy_accepted"] = True
            st.session_state["policy_declined"] = False
            st.session_state["play_intro"] = True
            st.query_params["consent"] = POLICY_VERSION
            st.rerun()


if st.query_params.get("consent") == POLICY_VERSION:
    st.session_state["policy_accepted"] = True

if (
    st.session_state.get("policy_declined", False)
    and not st.session_state.get("policy_accepted", False)
):
    st.markdown(
        '<div class="intro" style="padding-top:18vh"><div class="eyebrow">Your choice is respected</div><h2>SynapseAI remains closed.</h2><p>You can review the policy again whenever you are ready.</p></div>',
        unsafe_allow_html=True,
    )
    if st.button("Review terms and privacy policy", type="primary"):
        st.session_state["policy_declined"] = False
        st.rerun()
    st.stop()

if not st.session_state.get("policy_accepted", False):
    show_policy_dialog()
    st.stop()

if st.session_state.get("play_intro", False):
    components.html(
        f"""
        <script>
        // Production must also log consent server-side for an auditable record.
        localStorage.setItem("synapseai_consent", JSON.stringify({{
          accepted: true,
          timestamp: new Date().toISOString(),
          policyVersion: "{POLICY_VERSION}"
        }}));
        // A production redirect to the main application could be triggered here.
        </script>
        """,
        height=0,
    )


# --------------------------------------------------
# Dataset helpers for demo
# --------------------------------------------------

META_COLS = [
    "AIRC_ID",
    "ConstructName",
    "Wave",
    "HasData",
    "NumScores",
]

FLAG_COLS = [
    "Global",
    "Thickness",
    "Volume",
    "Area",
    "SubVolumes",
]

@st.cache_data
def load_demo_patient(
    patient_id=2561
):

    data_path = (
        PROJECT_ROOT
        / "dataset"
        / "BIOMARKERDATA.xlsx"
    )


    def load_sheet(sheet_name):

        df = pd.read_excel(
            data_path,
            sheet_name=sheet_name
        )

        drop_cols = [
            col
            for col in META_COLS + FLAG_COLS
            if col in df.columns
        ]

        return df.drop(
            columns=drop_cols
        )


    age_df = pd.read_excel(
        data_path,
        sheet_name="Age-W1,W2,W3"
    )

    age_df = age_df[
        ["S#", "AgeMRI_W1"]
    ].dropna(
        subset=["AgeMRI_W1"]
    )


    sheets = [
        "GlobalVariables-W1",
        "CorticalThickness-W1",
        "GMVolume-W1",
        "SurfaceArea-W1",
        "SubcorticalVolume-W1",
    ]


    merged = age_df.copy()


    for sheet in sheets:

        merged = merged.merge(
            load_sheet(sheet),
            on="S#",
            how="inner"
        )


    patient = merged[
        pd.to_numeric(
            merged["S#"],
            errors="coerce"
        ) == patient_id
    ]


    if patient.empty:

        raise ValueError(
            f"Demo patient "
            f"{patient_id} not found."
        )


    patient = patient.iloc[0]


    artifacts = load_model_artifacts()

    feature_cols = artifacts[
        "feature_cols"
    ]


    feature_values = {
        feature: float(
            patient[feature]
        )
        for feature in feature_cols
    }


    return {
        "patient_id":
            patient_id,

        "actual_age":
            float(
                patient["AgeMRI_W1"]
            ),

        "features":
            feature_values,
    }


# --------------------------------------------------
# Run complete pipeline
# --------------------------------------------------

def run_complete_pipeline(
    features,
    actual_age,
    patient_id=None,
):
    """
    MRI features
        ↓
    XGBoost
        ↓
    SHAP
        ↓
    Amass
        ↓
    Claude
    """

    with st.status(
        "Analyzing brain-age profile...",
        expanded=True,
    ) as status:

        st.write(
            "🧠 Running brain-age model..."
        )

        model_result = analyze_patient(
            feature_values=features,
            actual_age=actual_age,
            patient_id=patient_id,
            top_n=5,
        )


        st.write(
            "🔍 Identifying the five "
            "strongest contributing features..."
        )


        st.write(
            "📚 Searching scientific evidence "
            "with Amass..."
        )

        evidence_result = (
            build_evidence_from_model_result(
                model_result
            )
        )


        st.write(
            "✨ Translating the evidence "
            "into plain language with Claude..."
        )

        final_report = (
            build_patient_explanation(
                evidence_result
            )
        )

        # Claude supplies the plain-language explanation. Preserve the
        # numerical values already calculated by the model for the UI.
        final_report = attach_analysis_values(
            final_report,
            evidence_result,
        )


        status.update(
            label="Analysis complete",
            state="complete",
            expanded=False,
        )


    return final_report


# --------------------------------------------------
# Render report
# --------------------------------------------------

def render_friday_control(key, narration, title, description):
    trigger_label = f"friday-trigger-{key}"
    with st.container(key=f"friday_trigger_{key}"):
        triggered = st.button(trigger_label, key=f"friday_button_{key}")

    components.html(
        f"""
        <style>
        *{{box-sizing:border-box}} body{{margin:0;background:transparent;font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}}
        .guide{{width:100%;height:68px;display:flex;align-items:center;gap:13px;padding:10px 14px;border:1px solid rgba(81,215,208,.25);border-radius:16px;background:linear-gradient(110deg,rgba(20,40,49,.96),rgba(10,22,31,.94));color:#edf4f3;cursor:default;user-select:none;transition:.25s ease}}
        .guide:hover{{border-color:rgba(81,215,208,.55);transform:translateY(-2px)}}
        .orb{{position:relative;flex:none;width:34px;height:34px;border-radius:50%;background:radial-gradient(circle at 35% 30%,#d8ffff,#51d7d0 28%,#19717a 70%);box-shadow:0 0 0 7px rgba(81,215,208,.08),0 0 20px rgba(81,215,208,.42);animation:pulse 2.2s ease-in-out infinite}}
        .orb:after{{content:'';position:absolute;inset:-5px;border:1px solid rgba(81,215,208,.38);border-radius:50%;animation:ring 2.2s ease-out infinite}}
        strong{{display:block;font-size:13px;margin-bottom:3px}} span{{display:block;color:#91a3aa;font-size:11px}} .hint{{margin-left:auto;color:#51d7d0;font-size:10px;letter-spacing:.1em;text-transform:uppercase}}
        @keyframes pulse{{50%{{transform:scale(1.08)}}}} @keyframes ring{{0%{{transform:scale(.8);opacity:0}}40%{{opacity:.8}}100%{{transform:scale(1.35);opacity:0}}}}
        @media(prefers-reduced-motion:reduce){{.orb,.orb:after{{animation:none}}}}
        </style>
        <div class="guide" ondblclick="speak()" title="Double-click to hear Friday">
          <div class="orb"></div><div><strong>{escape(title)}</strong><span>{escape(description)}</span></div><div class="hint">Double-click</div>
        </div>
        <script>
        function speak(){{
          const target={json.dumps(trigger_label)};
          const button=[...window.parent.document.querySelectorAll('button')].find(el=>el.innerText.trim()===target);
          if(button) button.click();
        }}
        </script>
        """,
        height=72,
    )

    if triggered:
        try:
            with st.spinner("Friday is preparing your narration..."):
                st.session_state[f"friday_audio_{key}"] = load_friday_audio(
                    narration
                )
        except Exception as exc:
            st.error("Friday could not create this narration right now. Please try again.")
            print("ElevenLabs narration error:", repr(exc))

    audio = st.session_state.get(f"friday_audio_{key}")
    if audio:
        st.audio(audio, format="audio/mpeg", autoplay=True)


def render_report(report):
    patient = report["patient"]
    summary = report["summary"]
    factors = report["factors"]
    gap = float(patient["brain_age_gap"])

    st.markdown('<div class="report-shell"><div class="report-kicker">✦ Your personal brain-age story</div><div class="report-title">Your data, clearly explained.</div></div>', unsafe_allow_html=True)



    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Your age", f'{patient["actual_age"]:.0f} years')
    with col2:
        st.metric("Model estimate", f'{patient["predicted_age"]:.1f} years')
    with col3:
        st.metric("Difference", f"{abs(gap):.1f} years {'lower' if gap < 0 else 'higher'}")

    lower = min(patient["actual_age"], patient["predicted_age"]) - 8
    upper = max(patient["actual_age"], patient["predicted_age"]) + 8
    actual_position = 100 * (patient["actual_age"] - lower) / (upper - lower)
    predicted_position = 100 * (patient["predicted_age"] - lower) / (upper - lower)
    st.markdown(f'''<div class="insight-panel">
        <div class="age-track"><span class="age-marker" style="left:{predicted_position:.1f}%"></span><span class="age-marker" style="left:{actual_position:.1f}%;background:#ffba7a;box-shadow:0 0 0 3px #b47b4e"></span></div>
        <div class="age-labels"><span>Model estimate · {patient["predicted_age"]:.1f}</span><span>Your age · {patient["actual_age"]:.0f}</span></div>
    </div>''', unsafe_allow_html=True)

    st.markdown(f"### {summary['headline']}")
    st.write(summary["overview"])
    st.info("✦ " + summary["key_takeaway"])
    render_friday_control(
        "report_overview",
        build_report_narration(report),
        "Listen to your overview",
        "A calm, guided summary of your ages, overall pattern, and key takeaway.",
    )

    st.markdown('<div id="what-influenced-the-prediction" class="section-kicker">Why the model estimated this</div><div class="section-heading">What’s influencing the prediction?</div>', unsafe_allow_html=True)
    st.caption("Each measurement nudged the estimate higher or lower. The bar length shows how strongly the model leaned on it for this result.")

    impacts = [abs(float(f.get("shap_impact") or 0)) for f in factors]
    maximum = max(impacts, default=1) or 1
    rows = []
    for factor in factors:
        impact = float(factor.get("shap_impact") or 0)
        direction = factor["direction"]
        width = max(4, abs(impact) / maximum * 48)
        side = "lower" if direction == "younger" else "higher"
        rows.append(f'''<div class="contribution-row"><div class="contribution-head"><span>{escape(factor["feature"])}</span><span>{impact:+.2f} · pushes estimate {side}</span></div><div class="contribution-track"><span class="contribution-mid"></span><span class="contribution-bar {direction}" style="width:{width:.1f}%"></span></div></div>''')
    st.markdown('<div class="insight-panel">' + ''.join(rows) + '<div class="axis-labels"><span>← lowers estimated age</span><span>raises estimated age →</span></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Your biomarker context</div><div class="section-heading">Five measurements worth understanding</div>', unsafe_allow_html=True)
    cards = []
    for factor in factors:
        z_score = factor.get("z_score")
        notable = z_score is not None and abs(float(z_score)) >= 1
        if z_score is None:
            range_label = "Reference context unavailable"
        elif notable:
            range_label = "Unusually high" if float(z_score) > 0 else "Unusually low"
        else:
            range_label = "Within usual range"
        direction_label = "Younger-leaning" if factor["direction"] == "younger" else "Older-leaning"
        z_text = f" · z {float(z_score):+.2f}" if z_score is not None else ""
        feature_lower = factor["feature"].lower()
        icon = "◉" if "ventricle" in feature_lower else "⌁" if "white matter" in feature_lower else "◌"
        short_description = factor["what_it_is"].split(". ")[0].rstrip(".") + "."
        z_position = 50 if z_score is None else max(3, min(97, (float(z_score) + 2.5) / 5 * 100))
        cards.append(f'''<div class="biomarker-card">
            <div class="biomarker-top"><div style="display:flex;gap:.75rem;align-items:center"><div class="biomarker-icon">{icon}</div><div class="biomarker-name">{escape(factor["feature"])}</div></div><span class="biomarker-badge {'notable' if notable else ''}">{range_label}</span></div>
            <p class="biomarker-desc">{escape(short_description)}</p><div class="biomarker-meta">{direction_label}{z_text}</div>
            <div class="reference-mini"><span style="left:{z_position:.1f}%"></span></div><div class="reference-labels"><span>Below reference</span><span>Average</span><span>Above reference</span></div>
            <details><summary>＋ Why this matters</summary><p>{escape(factor["personal_interpretation"])}</p></details>
        </div>''')
    st.markdown('<div class="biomarker-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Visual analysis</div><div class="section-heading">Explore each biomarker</div><div class="carousel-hint">Swipe or scroll through all five analyses <span>→</span></div>', unsafe_allow_html=True)
    evidence_levels = {"insufficient": 1, "limited": 2, "moderate": 3, "strong": 4}
    analysis_cards = []
    for index, factor in enumerate(factors, start=1):
        z_score = factor.get("z_score")
        z_value = float(z_score) if z_score is not None else 0.0
        gauge_fraction = min(abs(z_value) / 2.5, 1.0)
        circumference = 339.3
        dash = circumference * gauge_fraction
        impact = float(factor.get("shap_impact") or 0)
        impact_width = max(5, abs(impact) / maximum * 100)
        direction = factor["direction"]
        direction_label = "Lowers age estimate" if direction == "younger" else "Raises age estimate"
        reference_label = "Above reference" if z_value >= 1 else "Below reference" if z_value <= -1 else "Near reference average"
        level = evidence_levels.get(factor["evidence_strength"].lower(), 1)
        dots = ''.join(f'<i class="{"on" if dot <= level else ""}"></i>' for dot in range(1, 5))
        analysis_cards.append(f'''<article class="analysis-card">
            <div class="analysis-index">◉ Analysis {index:02d} of 05</div>
            <h3>{escape(factor["feature"])}</h3><div class="analysis-subtitle">Personal MRI biomarker</div>
            <div class="mini-dashboard">
                <div class="z-gauge"><svg viewBox="0 0 128 128"><circle class="gauge-bg" cx="64" cy="64" r="54"/><circle class="gauge-progress" cx="64" cy="64" r="54" stroke-dasharray="{dash:.1f} {circumference:.1f}"/></svg><div class="gauge-center"><span class="gauge-number">{z_value:+.2f}</span><span class="gauge-label">reference z</span></div></div>
                <div><div class="mini-stat"><div class="mini-stat-label">Reference position</div><div class="mini-stat-value">{reference_label}</div></div><div class="mini-stat"><div class="mini-stat-label">Model direction</div><div class="mini-stat-value">{direction_label}</div></div></div>
            </div>
            <div class="mini-stat"><div class="mini-stat-label">Relative model contribution · {impact:+.2f}</div><div class="microbar {direction}"><span style="width:{impact_width:.1f}%"></span></div></div>
            <div class="mini-stat"><div class="mini-stat-label">Research confidence · {escape(factor["evidence_strength"].capitalize())}</div><div class="evidence-dots">{dots}</div></div>
            <p class="analysis-takeaway">{escape(factor["model_explanation"])}</p>
        </article>''')
    st.markdown('<div class="analysis-carousel">' + ''.join(analysis_cards) + '</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Explore each signal</div><div class="section-heading">The detail behind your result</div>', unsafe_allow_html=True)
    detail_columns = st.columns(2)
    for index, factor in enumerate(factors, start=1):
        direction_label = "Younger-leaning" if factor["direction"] == "younger" else "Older-leaning"
        strength = factor["evidence_strength"].capitalize()
        with detail_columns[(index - 1) % 2]:
            with st.expander(f"{index:02d}  ·  {factor['feature']}"):
                st.markdown(f'<div class="factor-heading"><span class="factor-pill">{direction_label}</span><span class="factor-pill">{strength} evidence</span></div>', unsafe_allow_html=True)
                st.markdown("#### What this is")
                st.write(factor["what_it_is"])
                st.markdown("#### What the model saw")
                st.write(factor["model_explanation"])
                st.markdown("#### What it means in your result")
                st.write(factor["personal_interpretation"])
                st.markdown("#### What published research adds")
                st.write(factor["research_summary"])
                st.caption(factor["evidence_note"])

                sources = factor.get("sources", [])
                if sources:
                    st.markdown("#### Amass research sources")
                    for source in sources:
                        title, doi = source["title"], source.get("doi")
                        st.markdown(f"- [{title}](https://doi.org/{doi})" if doi else f"- {title}")

                render_friday_control(
                    f"factor_{index}",
                    build_factor_narration(factor),
                    "Listen with Friday",
                    "Hear this biomarker, its model influence, and its Amass research context as one natural explanation.",
                )

    st.markdown("## How strong is the research context?")
    st.write(summary["evidence_overview"])
    st.markdown('''<div class="disclaimer"><strong>Your results, made understandable</strong><br>Synapse AI brings your brain-age estimate, influential measurements, and Amass research context into one personalized explanation you can explore at your own pace.</div>''', unsafe_allow_html=True)


# --------------------------------------------------
# HERO
# --------------------------------------------------

play_intro = st.session_state.pop("play_intro", False)
hero_class = "hero intro-play" if play_intro else "hero"

if play_intro:
    st.toast("Policy accepted. Welcome to SynapseAI.", icon="✅")
    st.markdown(
        """
        <div class="intro-transition" aria-hidden="true">
            <div class="intro-brand">Synapse<span>AI</span></div>
            <svg class="intro-brain" viewBox="0 0 620 620">
                <path d="M310 74 C224 66 158 98 121 151 C91 188 78 231 84 271 C64 305 69 353 94 380 C94 456 137 507 206 536 C232 571 268 588 310 588 C353 588 390 570 416 535 C485 506 527 453 526 376 C549 344 552 301 532 266 C535 224 519 181 488 145 C447 96 387 67 310 74Z"/>
                <path d="M310 84 C312 215 294 410 310 588 M142 173 C205 224 207 297 139 345 M478 166 C418 220 419 291 490 338 M121 405 C205 424 244 476 247 528 M500 397 C416 418 371 470 367 528 M215 264 C270 274 288 318 249 360 M408 256 C356 272 339 315 380 354" fill="none"/>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
<div class="topbar">
    <div class="wordmark">Synapse <span>AI</span><span class="pill">AI-guided · educational</span></div>
    <div class="nav-note">Explainable brain-age intelligence</div>
</div>
<div class="{hero_class}">
    <div class="hero-copy">
        <div class="eyebrow">✧ &nbsp; Narrated longitudinal brain-age analysis</div>
        <h1>The Earlier You Understand Your Own Aging, the More Say You Get in It.</h1>
        <p>Aging is invisible until it isn't. We read your brain's current biomarkers and turn them into a plain-language, personal head start—so you understand your own data sooner, while there's still time to act.</p>
        <div class="hero-actions">
            <a class="hero-button primary" href="#try-neurodecel">Start brain analysis</a>
            <a class="hero-button" href="#how-it-works">See what influenced the model</a>
        </div>
    </div>
    <div class="brain-visual" aria-label="Animated MRI brain measurement illustration">
        <svg viewBox="0 0 620 660" role="img">
            <defs>
                <radialGradient id="brainFill" cx="45%" cy="45%" r="65%"><stop offset="0" stop-color="#203f52" stop-opacity=".9"/><stop offset="1" stop-color="#152739" stop-opacity=".72"/></radialGradient>
            </defs>
            <circle class="scan-ring" cx="310" cy="310" r="275"/><circle class="scan-ring" cx="310" cy="310" r="232"/>
            <path class="brain-outline" d="M310 84 C232 77 168 104 133 150 C105 183 91 220 94 260 C76 288 78 333 100 360 C99 432 137 482 205 511 C231 548 268 568 310 568 C353 568 391 548 416 511 C484 482 522 431 521 356 C543 328 544 285 526 256 C527 214 513 177 485 145 C447 101 386 78 310 84Z"/>
            <path class="brain-stem" d="M290 540 C292 590 278 619 255 643"/>
            <path class="brain-detail" d="M310 92 C311 204 295 386 310 568 M151 175 C205 220 205 286 144 326 M472 166 C420 217 422 280 482 319 M128 388 C205 405 246 460 248 500 M486 379 C410 400 367 451 366 501 M222 255 C268 266 284 302 250 341 M399 248 C357 263 344 304 379 337"/>
            <path class="brain-circuit" d="M118 220 L174 156 L286 218 L379 164 L462 235 L485 324 L407 406 L310 316 L235 303 L158 354 L205 447 L258 530 L310 432 L400 454"/>
            <path class="brain-circuit" d="M174 156 L286 218 L250 303 M286 218 L379 164 M310 316 L400 454 M310 316 L258 530"/>
            <g class="node-hotspot"><circle class="brain-node" cx="118" cy="220" r="7"/><foreignObject x="132" y="176" width="190" height="55"><div class="node-popup">Superior temporal cortex thickness</div></foreignObject></g>
            <circle class="brain-node dim" cx="174" cy="156" r="6"/><circle class="brain-node" cx="286" cy="218" r="6"/>
            <g class="node-hotspot"><circle class="brain-node" cx="405" cy="173" r="7"/><foreignObject x="220" y="120" width="175" height="55"><div class="node-popup">Third ventricle volume</div></foreignObject></g>
            <g class="node-hotspot"><circle class="brain-node" cx="462" cy="235" r="7"/><foreignObject x="275" y="190" width="175" height="55"><div class="node-popup">White matter hypointensity volume</div></foreignObject></g>
            <circle class="brain-node dim" cx="485" cy="324" r="6"/><circle class="brain-node" cx="310" cy="316" r="7"/><circle class="brain-node dim" cx="250" cy="303" r="6"/>
            <g class="node-hotspot"><circle class="brain-node" cx="158" cy="354" r="7"/><foreignObject x="172" y="310" width="190" height="55"><div class="node-popup">Non-white matter hypointensity volume</div></foreignObject></g>
            <circle class="brain-node dim" cx="407" cy="406" r="6"/><circle class="brain-node dim" cx="205" cy="447" r="6"/><circle class="brain-node" cx="258" cy="530" r="6"/><circle class="brain-node dim" cx="310" cy="432" r="6"/>
            <g class="node-hotspot"><circle class="brain-node" cx="400" cy="454" r="6"/><foreignObject x="210" y="408" width="180" height="55"><div class="node-popup">Pars triangularis cortical thickness</div></foreignObject></g>
            <circle class="target-ring" cx="205" cy="447" r="34"/><circle class="brain-node" cx="205" cy="447" r="12"/>
            <circle class="brain-node dim" cx="75" cy="505" r="5"/><circle class="brain-node" cx="548" cy="325" r="4"/><circle class="brain-node dim" cx="529" cy="478" r="4"/>
        </svg>
        <div class="brain-tag">MRI-derived measurements · 270 features</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if "friday_welcome_narration" not in st.session_state:
    st.session_state["friday_welcome_narration"] = build_welcome_narration()

render_friday_control(
    "welcome",
    st.session_state["friday_welcome_narration"],
    "Friday",
    "Your quiet voice guide. Double-click when you would rather listen.",
)


st.markdown(
    """
    <div class="intro" id="how-it-works">
        <div class="eyebrow">A clearer kind of result</div>
        <h2>Your results, explained for you.</h2>
        <p>Plain-language context grounded in your own data—so you can understand the numbers, see what shaped them, and learn what relevant research adds without having to piece it together from a search bar.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Product value
# --------------------------------------------------

c1, c2, c3 = st.columns(3)


with c1:
    st.markdown('<div class="feature-card"><div class="feature-number">01 · UNDERSTAND</div><h3>See your brain-age estimate</h3><p>Bring hundreds of MRI-derived measurements together into one clear, personal view.</p></div>', unsafe_allow_html=True)


with c2:
    st.markdown('<div class="feature-card"><div class="feature-number">02 · EXPLAIN</div><h3>See what shaped your result</h3><p>Understand which of your measurements influenced the estimate and in which direction.</p></div>', unsafe_allow_html=True)


with c3:
    st.markdown('<div class="feature-card"><div class="feature-number">03 · CONTEXTUALIZE</div><h3>Connect your data to research</h3><p>Explore relevant published evidence translated into calm, understandable language.</p></div>', unsafe_allow_html=True)


st.divider()


# --------------------------------------------------
# Input choices
# --------------------------------------------------

st.markdown('<div id="try-neurodecel" class="section-kicker">Begin an analysis</div><div class="section-heading">Try Synapse AI</div>', unsafe_allow_html=True)


upload_tab, demo_tab = st.tabs(
    [
        "📄 Upload MRI report",
        "⚡ Try a demo",
    ]
)


# --------------------------------------------------
# PDF tab
# --------------------------------------------------

with upload_tab:

    st.markdown(
        "### Upload an MRI-derived brain report"
    )

    st.write(
        """
        Upload a structured PDF containing your
        chronological age and MRI-derived structural
        measurements.

        Synapse AI brings those measurements together,
        estimates brain age, shows what shaped the result,
        and adds relevant research context in plain language.
        """
    )


    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        key="patient_pdf",
    )


    if uploaded_file:

        st.success(
            f"Loaded: {uploaded_file.name}"
        )


        # ------------------------------------------
        # First parse + validate PDF
        # ------------------------------------------

        try:

            with st.spinner(
                "Checking MRI report..."
            ):

                parsed = parse_patient_pdf(
                    uploaded_file
                )


            st.markdown(
                "#### Report successfully parsed"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Chronological age",
                    f"{parsed['actual_age']:.0f}"
                )


            with col2:

                st.metric(
                    "MRI features detected",
                    f"{parsed['feature_count']} / 270"
                )


            st.success(
                "All required MRI measurements "
                "were found. Ready for analysis."
            )


            # ------------------------------------------
            # Full live analysis
            # ------------------------------------------

            if st.button(
                "🧠 Analyze brain age",
                type="primary",
                use_container_width=True,
            ):

                report = run_complete_pipeline(

                    features=
                        parsed["features"],

                    actual_age=
                        parsed["actual_age"],

                    patient_id=
                        parsed["patient_id"],
                )


                st.session_state[
                    "report"
                ] = report


        except Exception as exc:

            st.error(
                "This PDF could not be used "
                "for brain-age analysis."
            )

            st.warning(
                str(exc)
            )

            print(
                "PDF parsing error:",
                repr(exc)
            )


# --------------------------------------------------
# Demo tab
# --------------------------------------------------

with demo_tab:

    st.write(
        "Explore Synapse AI using an anonymized "
        "sample from our research dataset."
    )

    st.caption(
        "Demo patient: anonymized participant #2561"
    )

    if st.button(
        "🚀 Run demo",
        type="primary",
        use_container_width=False,
    ):

        try:

            report = load_cached_demo_report()

            st.session_state[
                "report"
            ] = report

            st.success(
                "Demo loaded successfully."
            )

        except Exception as exc:

            st.error(
                "Could not load the demo patient."
            )

            print(
                "Demo loading error:",
                repr(exc)
            )

# --------------------------------------------------
# Render persisted result
# --------------------------------------------------

if "report" in st.session_state:

    render_report(
        st.session_state[
            "report"
        ]
    )


# --------------------------------------------------
# Future vision
# --------------------------------------------------

st.divider()

st.markdown('<div class="section-kicker">What comes next</div><div class="section-heading">From MRI scan to understandable insight</div>', unsafe_allow_html=True)

st.write(
    """
    Today's prototype works with structural measurements
    already extracted from MRI scans.

    The next step is direct MRI ingestion: upload a scan,
    automatically extract structural brain measurements,
    estimate brain age, identify the most influential
    regions, and generate an evidence-grounded explanation
    — all in one workflow.
    """
)

st.caption(
    "Synapse AI is a personalized guide to understanding your "
    "brain-age estimate, your MRI-derived measurements, and the "
    "research context around them."
)
