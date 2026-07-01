# ==========================================================
# Lab Report OCR System — Ultra Futuristic UI v2
# ----------------------------------------------------------
# Thin Streamlit "view" layer. All compute lives in dedicated,
# unit-testable modules (config / logging_setup / cleanup /
# detect / ocr / preprocess / pdf_utils / pipeline / report /
# exports). This file only wires those modules to the UI and
# renders the futuristic interface.
# ==========================================================

import os
import sys
import io
from datetime import datetime

# Guarantee sibling modules resolve regardless of how Streamlit
# is launched (streamlit run app/app.py, or from within app/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Project modules (compute layer) ──
import config as config_mod
import logging_setup as logging_mod
import cleanup as cleanup_mod
import ocr as ocr_mod
import pdf_utils as pdf_mod
import exports as exports_mod
from detect import model_info
from pipeline import process_one, bundle_log_record
from report import summarise_fields

# Ultralytics is heavy; import defensively so the rest of the app
# (and clear error messaging) still works if it is missing.
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:  # pragma: no cover - import-time environment issue
    YOLO = None
    YOLO_AVAILABLE = False


# ----------------------------------------------------------
# Bootstrap — config, logging, housekeeping (runs once)
# ----------------------------------------------------------

@st.cache_resource(show_spinner=False)
def bootstrap():
    """Load config, configure logging, ensure dirs, clean temp — once."""
    cfg = config_mod.load_config("config.yaml")
    config_mod.ensure_dirs(cfg)
    logger = logging_mod.get_logger(cfg)
    removed = cleanup_mod.clean_old_temp(
        cfg.paths.temp_dir,
        max_age_hours=getattr(cfg.cleanup, "temp_max_age_hours", 24),
        logger=logger,
    )
    logger.info("App bootstrap complete (temp files removed: %d)", removed)
    return cfg, logger


cfg, logger = bootstrap()


@st.cache_resource(show_spinner=False)
def load_model(model_path):
    """Load and cache the YOLO model. Returns (model, info) or raises."""
    model = YOLO(model_path)
    return model, model_info(model, model_path)


@st.cache_resource(show_spinner=False)
def get_easyocr_reader(languages_tuple, gpu):
    """Build and cache an EasyOCR reader (languages passed as a tuple key)."""
    return ocr_mod.make_easyocr_reader(list(languages_tuple), gpu=gpu)


# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title=cfg.app.title,
    page_icon=cfg.app.page_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------
# Ultra Futuristic Dark Theme — Custom CSS
# ----------------------------------------------------------

st.markdown("""
<style>
    /* ── Import Modern Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-void: #050508;
        --bg-primary: #08080e;
        --bg-secondary: #0e0e18;
        --bg-card: rgba(14, 14, 28, 0.55);
        --bg-glass: rgba(255, 255, 255, 0.02);
        --border-glass: rgba(255, 255, 255, 0.05);
        --accent-cyan: #00e5ff;
        --accent-blue: #3b82f6;
        --accent-purple: #a855f7;
        --accent-violet: #7c3aed;
        --accent-pink: #ec4899;
        --accent-green: #10b981;
        --accent-amber: #f59e0b;
        --accent-red: #ef4444;
        --text-primary: #f0f4f8;
        --text-secondary: #8b95a5;
        --text-muted: #505868;
        --glow-cyan: 0 0 30px rgba(0, 229, 255, 0.25), 0 0 60px rgba(0, 229, 255, 0.1);
        --glow-purple: 0 0 30px rgba(168, 85, 247, 0.25), 0 0 60px rgba(168, 85, 247, 0.1);
        --glow-pink: 0 0 30px rgba(236, 72, 153, 0.2);
    }

    /* ── Global Reset ── */
    .stApp {
        background: var(--bg-void) !important;
        font-family: 'Outfit', sans-serif !important;
        overflow-x: hidden;
    }

    /* ── Animated Aurora Background ── */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background:
            radial-gradient(ellipse 120% 80% at 15% 10%, rgba(0, 229, 255, 0.06) 0%, transparent 55%),
            radial-gradient(ellipse 80% 120% at 85% 90%, rgba(168, 85, 247, 0.06) 0%, transparent 55%),
            radial-gradient(ellipse 60% 60% at 50% 40%, rgba(236, 72, 153, 0.03) 0%, transparent 50%),
            radial-gradient(ellipse 40% 40% at 70% 20%, rgba(59, 130, 246, 0.04) 0%, transparent 40%);
        pointer-events: none;
        z-index: 0;
        animation: auroraShift 15s ease-in-out infinite alternate;
    }

    @keyframes auroraShift {
        0% { filter: hue-rotate(0deg) brightness(1); }
        50% { filter: hue-rotate(15deg) brightness(1.1); }
        100% { filter: hue-rotate(-10deg) brightness(1); }
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink));
        border-radius: 10px;
    }

    /* ── Typography ── */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text-primary) !important;
        font-family: 'Outfit', sans-serif !important;
    }

    p, span, label, .stMarkdown p {
        color: var(--text-secondary) !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* ── Hide Streamlit Chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ═════════════════════════════════════════════════════
       HERO SECTION
    ═════════════════════════════════════════════════════ */

    .hero-wrapper {
        position: relative;
        text-align: center;
        padding: 3.5rem 2rem 2.5rem;
        overflow: hidden;
    }

    /* Orbit rings behind hero */
    .hero-wrapper::before,
    .hero-wrapper::after {
        content: '';
        position: absolute;
        border-radius: 50%;
        border: 1px solid rgba(0, 229, 255, 0.06);
        pointer-events: none;
    }

    .hero-wrapper::before {
        width: 600px; height: 600px;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        animation: orbitSpin 30s linear infinite;
    }

    .hero-wrapper::after {
        width: 400px; height: 400px;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        border-color: rgba(168, 85, 247, 0.06);
        animation: orbitSpin 20s linear infinite reverse;
    }

    @keyframes orbitSpin {
        from { transform: translate(-50%, -50%) rotate(0deg); }
        to { transform: translate(-50%, -50%) rotate(360deg); }
    }

    .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 1.2rem;
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.08), rgba(168, 85, 247, 0.08));
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-radius: 100px;
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--accent-cyan) !important;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 1.8rem;
        position: relative;
        z-index: 1;
        animation: chipGlow 3s ease-in-out infinite alternate;
    }

    @keyframes chipGlow {
        0% { box-shadow: 0 0 10px rgba(0, 229, 255, 0.1); }
        100% { box-shadow: 0 0 25px rgba(0, 229, 255, 0.2), 0 0 50px rgba(0, 229, 255, 0.05); }
    }

    .hero-title {
        font-size: 3.8rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        line-height: 1.05;
        margin-bottom: 0.6rem;
        position: relative;
        z-index: 1;
        background: linear-gradient(135deg, #ffffff 0%, #00e5ff 30%, #a855f7 60%, #ec4899 85%, #ffffff 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientFlow 6s ease-in-out infinite;
    }

    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-sub {
        font-size: 1.05rem;
        color: var(--text-muted) !important;
        font-weight: 400;
        max-width: 560px;
        margin: 0 auto;
        line-height: 1.7;
        position: relative;
        z-index: 1;
    }

    /* ── Animated Neon Divider ── */
    .neon-line {
        height: 1px;
        border: none;
        margin: 1.5rem 0 2rem;
        background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-purple), var(--accent-pink), transparent);
        background-size: 200% 100%;
        animation: neonSlide 4s linear infinite;
        opacity: 0.6;
    }

    @keyframes neonSlide {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    /* ═════════════════════════════════════════════════════
       HOLOGRAPHIC GLASS CARDS
    ═════════════════════════════════════════════════════ */

    .holo-card {
        position: relative;
        background: linear-gradient(145deg, rgba(14, 14, 28, 0.7), rgba(8, 8, 14, 0.8));
        backdrop-filter: blur(24px) saturate(1.2);
        -webkit-backdrop-filter: blur(24px) saturate(1.2);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .holo-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.03) 0%, transparent 50%, rgba(168, 85, 247, 0.03) 100%);
        pointer-events: none;
    }

    /* Animated top-border glow */
    .holo-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-purple), transparent);
        background-size: 200% 100%;
        animation: borderGlow 3s linear infinite;
        opacity: 0;
        transition: opacity 0.4s ease;
    }

    .holo-card:hover::after {
        opacity: 1;
    }

    .holo-card:hover {
        border-color: rgba(0, 229, 255, 0.12);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), 0 0 30px rgba(0, 229, 255, 0.06);
        transform: translateY(-3px);
    }

    @keyframes borderGlow {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    /* ── Card Headers ── */
    .card-head {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 1.2rem;
    }

    .card-ico {
        width: 44px; height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        position: relative;
    }

    .card-ico.ico-cyan {
        background: rgba(0, 229, 255, 0.08);
        box-shadow: inset 0 0 20px rgba(0, 229, 255, 0.05);
    }
    .card-ico.ico-purple {
        background: rgba(168, 85, 247, 0.08);
        box-shadow: inset 0 0 20px rgba(168, 85, 247, 0.05);
    }
    .card-ico.ico-pink {
        background: rgba(236, 72, 153, 0.08);
        box-shadow: inset 0 0 20px rgba(236, 72, 153, 0.05);
    }
    .card-ico.ico-green {
        background: rgba(16, 185, 129, 0.08);
        box-shadow: inset 0 0 20px rgba(16, 185, 129, 0.05);
    }

    .card-lbl {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary) !important;
        letter-spacing: -0.01em;
    }

    .card-sub {
        font-size: 0.8rem;
        color: var(--text-muted) !important;
        line-height: 1.5;
        margin-top: 0.1rem;
    }

    /* ═════════════════════════════════════════════════════
       STATUS INDICATORS
    ═════════════════════════════════════════════════════ */

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.55rem 1.2rem;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    .pill-ok {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(16, 185, 129, 0.04));
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #10b981 !important;
    }

    .pill-err {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0.04));
        border: 1px solid rgba(239, 68, 68, 0.2);
        color: #ef4444 !important;
    }

    .pill-info {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.08), rgba(0, 229, 255, 0.04));
        border: 1px solid rgba(0, 229, 255, 0.2);
        color: var(--accent-cyan) !important;
    }

    /* Live dot */
    .live-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--accent-green);
        position: relative;
    }

    .live-dot::after {
        content: '';
        position: absolute;
        top: -3px; left: -3px;
        width: 14px; height: 14px;
        border-radius: 50%;
        background: rgba(16, 185, 129, 0.3);
        animation: livePulse 2s ease-in-out infinite;
    }

    @keyframes livePulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.8); opacity: 0; }
    }

    /* ═════════════════════════════════════════════════════
       STATS GRID
    ═════════════════════════════════════════════════════ */

    .stats-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1.5rem 0 0.5rem;
    }

    .s-card {
        position: relative;
        background: linear-gradient(145deg, rgba(14, 14, 28, 0.6), rgba(8, 8, 14, 0.7));
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        overflow: hidden;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .s-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: radial-gradient(circle at 50% 0%, rgba(0, 229, 255, 0.04) 0%, transparent 60%);
        pointer-events: none;
    }

    .s-card:hover {
        transform: translateY(-4px) scale(1.02);
        border-color: rgba(0, 229, 255, 0.15);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3), 0 0 20px rgba(0, 229, 255, 0.05);
    }

    .s-num {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace !important;
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-purple) 50%, var(--accent-pink) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }

    .s-lbl {
        font-size: 0.65rem;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-top: 0.4rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* ═════════════════════════════════════════════════════
       FILE UPLOADER
    ═════════════════════════════════════════════════════ */

    [data-testid="stFileUploader"] {
        background: linear-gradient(145deg, rgba(14, 14, 28, 0.5), rgba(8, 8, 14, 0.6)) !important;
        border: 2px dashed rgba(0, 229, 255, 0.12) !important;
        border-radius: 18px !important;
        padding: 1.5rem !important;
        transition: all 0.4s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0, 229, 255, 0.3) !important;
        background: rgba(0, 229, 255, 0.02) !important;
        box-shadow: 0 0 30px rgba(0, 229, 255, 0.04) !important;
    }

    [data-testid="stFileUploader"] label p {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploader"] small {
        color: var(--text-muted) !important;
    }

    /* ═════════════════════════════════════════════════════
       BUTTONS
    ═════════════════════════════════════════════════════ */

    .stButton > button {
        position: relative;
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.1), rgba(168, 85, 247, 0.1), rgba(236, 72, 153, 0.05)) !important;
        color: var(--text-primary) !important;
        border: 1px solid rgba(0, 229, 255, 0.25) !important;
        border-radius: 14px !important;
        padding: 0.85rem 2.5rem !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1rem !important;
        letter-spacing: 0.03em !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
        overflow: hidden !important;
    }

    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important; left: -100% !important;
        width: 100% !important; height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(0, 229, 255, 0.08), transparent) !important;
        transition: left 0.5s ease !important;
    }

    .stButton > button:hover::before {
        left: 100% !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.18), rgba(168, 85, 247, 0.18), rgba(236, 72, 153, 0.1)) !important;
        border-color: var(--accent-cyan) !important;
        box-shadow: var(--glow-cyan), 0 10px 30px rgba(0, 0, 0, 0.3) !important;
        transform: translateY(-3px) !important;
    }

    .stButton > button:active {
        transform: translateY(-1px) !important;
    }

    /* ═════════════════════════════════════════════════════
       SLIDER & TOGGLE
    ═════════════════════════════════════════════════════ */

    .stSlider > div > div { color: var(--text-secondary) !important; }

    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: var(--text-muted) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ═════════════════════════════════════════════════════
       IMAGES
    ═════════════════════════════════════════════════════ */

    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.04);
        transition: all 0.3s ease;
    }

    [data-testid="stImage"]:hover {
        border-color: rgba(0, 229, 255, 0.1);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }

    /* ═════════════════════════════════════════════════════
       INFO PANEL
    ═════════════════════════════════════════════════════ */

    .info-grid {
        display: grid;
        gap: 0;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.85rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }

    .info-row:last-child { border-bottom: none; }

    .info-key {
        font-size: 0.7rem;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    .info-val {
        font-size: 0.88rem;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 500;
    }

    /* ═════════════════════════════════════════════════════
       SCANNING LINE EFFECT ON RESULT IMAGE
    ═════════════════════════════════════════════════════ */

    .scan-wrapper {
        position: relative;
        overflow: hidden;
        border-radius: 16px;
    }

    .scan-wrapper::after {
        content: '';
        position: absolute;
        top: -100%; left: 0;
        width: 100%; height: 3px;
        background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-purple), transparent);
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.5), 0 0 60px rgba(0, 229, 255, 0.2);
        animation: scanLine 3s ease-in-out infinite;
        pointer-events: none;
        z-index: 10;
    }

    @keyframes scanLine {
        0% { top: -2%; }
        100% { top: 102%; }
    }

    /* ═════════════════════════════════════════════════════
       EMPTY STATE
    ═════════════════════════════════════════════════════ */

    .empty-state {
        text-align: center;
        padding: 5rem 2rem;
        position: relative;
    }

    .empty-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        display: inline-block;
        animation: emptyFloat 4s ease-in-out infinite;
        filter: grayscale(0.3);
    }

    @keyframes emptyFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-12px); }
    }

    .empty-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary) !important;
        margin-bottom: 0.5rem;
    }

    .empty-desc {
        font-size: 0.9rem;
        color: var(--text-muted) !important;
        max-width: 420px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ═════════════════════════════════════════════════════
       FOOTER
    ═════════════════════════════════════════════════════ */

    .futr-footer {
        text-align: center;
        padding: 2.5rem 1rem;
        margin-top: 3rem;
        position: relative;
    }

    .futr-footer::before {
        content: '';
        position: absolute;
        top: 0; left: 10%;
        width: 80%; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 229, 255, 0.15), rgba(168, 85, 247, 0.15), transparent);
    }

    .footer-text {
        font-size: 0.72rem;
        color: var(--text-muted) !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600;
    }

    .footer-brand {
        font-size: 0.65rem;
        color: rgba(80, 88, 104, 0.5) !important;
        margin-top: 0.4rem;
        letter-spacing: 0.15em;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ═════════════════════════════════════════════════════
       OCR RESULTS TABLE
    ═════════════════════════════════════════════════════ */

    .ocr-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 16px;
        overflow: hidden;
        background: linear-gradient(145deg, rgba(14, 14, 28, 0.7), rgba(8, 8, 14, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.04);
    }

    .ocr-table thead th {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.06), rgba(168, 85, 247, 0.04));
        color: var(--accent-cyan) !important;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700;
        padding: 1rem 1.2rem;
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        font-family: 'Space Grotesk', sans-serif !important;
    }

    .ocr-table tbody td {
        padding: 0.9rem 1.2rem;
        font-size: 0.88rem;
        color: var(--text-secondary) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.02);
        font-family: 'Outfit', sans-serif !important;
        vertical-align: middle;
    }

    .ocr-table tbody tr:last-child td { border-bottom: none; }

    .ocr-table tbody tr {
        transition: all 0.25s ease;
    }

    .ocr-table tbody tr:hover td {
        background: rgba(0, 229, 255, 0.03);
    }

    .field-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace !important;
        background: rgba(168, 85, 247, 0.1);
        color: var(--accent-purple) !important;
        border: 1px solid rgba(168, 85, 247, 0.2);
    }

    .ocr-text {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem;
        color: var(--text-primary) !important;
        font-weight: 500;
        word-break: break-word;
    }

    .conf-mini {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .conf-high { color: var(--accent-green) !important; }
    .conf-mid { color: var(--accent-amber) !important; }
    .conf-low { color: var(--accent-pink) !important; }

    /* ═════════════════════════════════════════════════════
       CROP GALLERY
    ═════════════════════════════════════════════════════ */

    .crop-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .crop-item {
        position: relative;
        background: linear-gradient(145deg, rgba(14, 14, 28, 0.6), rgba(8, 8, 14, 0.7));
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 14px;
        overflow: hidden;
        transition: all 0.35s ease;
    }

    .crop-item:hover {
        border-color: rgba(0, 229, 255, 0.15);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }

    .crop-label {
        padding: 0.6rem 0.8rem;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--accent-cyan) !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'Space Grotesk', sans-serif !important;
        border-top: 1px solid rgba(255, 255, 255, 0.03);
        background: rgba(0, 229, 255, 0.03);
    }

    /* ═════════════════════════════════════════════════════
       DOWNLOAD SECTION
    ═════════════════════════════════════════════════════ */

    .dl-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.4rem; }
        .stats-row { grid-template-columns: repeat(2, 1fr); }
        .crop-gallery { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); }
        .hero-wrapper::before, .hero-wrapper::after { display: none; }
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# Floating Particles Background (Canvas)
# ----------------------------------------------------------

st.markdown("""
<canvas id="particles-bg" style="position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:0;pointer-events:none;"></canvas>
<script>
(function() {
    const c = document.getElementById('particles-bg');
    if (!c) return;
    const ctx = c.getContext('2d');
    let w, h, particles = [];

    function resize() {
        w = c.width = window.innerWidth;
        h = c.height = window.innerHeight;
    }

    function init() {
        resize();
        particles = [];
        const count = Math.min(60, Math.floor(w * h / 18000));
        for (let i = 0; i < count; i++) {
            particles.push({
                x: Math.random() * w,
                y: Math.random() * h,
                r: Math.random() * 1.5 + 0.3,
                dx: (Math.random() - 0.5) * 0.3,
                dy: (Math.random() - 0.5) * 0.3,
                o: Math.random() * 0.4 + 0.1,
                hue: Math.random() > 0.5 ? 185 : 270
            });
        }
    }

    function draw() {
        ctx.clearRect(0, 0, w, h);
        particles.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${p.hue}, 80%, 65%, ${p.o})`;
            ctx.fill();
            p.x += p.dx;
            p.y += p.dy;
            if (p.x < 0 || p.x > w) p.dx *= -1;
            if (p.y < 0 || p.y > h) p.dy *= -1;
        });

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(0, 229, 255, ${0.04 * (1 - dist/120)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }

    window.addEventListener('resize', resize);
    init();
    draw();
})();
</script>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# Hero Header
# ----------------------------------------------------------

st.markdown("""
<div class="hero-wrapper">
    <div class="hero-chip">◆ &nbsp;Neural Vision Engine &nbsp;•&nbsp; v2.0</div>
    <div class="hero-title">Lab Report OCR</div>
    <p class="hero-sub">
        Upload a laboratory report and let our YOLO neural network
        detect, classify, and extract structured data — in real time.
    </p>
</div>
<div class="neon-line"></div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# Small render helpers (keep the main flow readable)
# ----------------------------------------------------------

def pill(kind, html):
    """Render a status pill. ``kind`` is one of ok / err / info."""
    st.markdown(
        f'<div class="status-pill pill-{kind}">{html}</div>',
        unsafe_allow_html=True,
    )


def card_head(ico_class, icon, label, sub):
    """Render a futuristic section header card."""
    st.markdown(f"""
    <div class="card-head">
        <div class="card-ico {ico_class}">{icon}</div>
        <div>
            <div class="card-lbl">{label}</div>
            <div class="card-sub">{sub}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def stats_row(cards):
    """Render up to four (value, label) stat cards."""
    cells = "".join(
        f'<div class="s-card"><div class="s-num">{val}</div>'
        f'<div class="s-lbl">{lbl}</div></div>'
        for val, lbl in cards
    )
    st.markdown(f'<div class="stats-row">{cells}</div>', unsafe_allow_html=True)


def neon():
    st.markdown('<div class="neon-line"></div>', unsafe_allow_html=True)


def png_bytes(img_rgb):
    """Encode an RGB ndarray to PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img_rgb)).save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------
# Load YOLO Model + Model Information Panel
# ----------------------------------------------------------

MODEL_PATH = cfg.model.path

if not YOLO_AVAILABLE:
    pill("err", "✕ &nbsp;Ultralytics is not installed — run <code>pip install ultralytics</code>")
    st.stop()

if not os.path.exists(MODEL_PATH):
    pill("err", f"✕ &nbsp;Model not found at <code>{MODEL_PATH}</code>")
    logger.error("Model file missing at %s", MODEL_PATH)
    st.stop()

try:
    model, minfo = load_model(MODEL_PATH)
    pill("ok", "<span class='live-dot'></span> &nbsp;YOLO Model Online — Ready for Detection")
except Exception as exc:
    pill("err", f"✕ &nbsp;Error loading model: {exc}")
    logger.exception("Failed to load model")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("🧠  Model Information", expanded=False):
    classes_str = ", ".join(minfo.get("classes", [])) or "—"
    info_rows = [
        ("Model", minfo.get("name", "—")),
        ("Path", minfo.get("path", "—")),
        ("Task", minfo.get("task", "—")),
        ("Classes", str(minfo.get("num_classes", 0))),
        ("Ultralytics", minfo.get("version", "—")),
        ("Default Conf.", f"{cfg.model.default_confidence:.2f}"),
    ]
    rows_html = "".join(
        f'<div class="info-row"><span class="info-key">{k}</span>'
        f'<span class="info-val">{v}</span></div>'
        for k, v in info_rows
    )
    st.markdown(f"""
    <div class="holo-card">
        <div class="card-lbl" style="margin-bottom: 1rem; font-size: 0.9rem;">🧠 Neural Model</div>
        <div class="info-grid">{rows_html}</div>
        <div class="card-sub" style="margin-top: 1rem;">
            <strong style="color: var(--accent-cyan);">Detected fields:</strong> {classes_str}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ----------------------------------------------------------
# Layout — Upload + Settings
# ----------------------------------------------------------

col_upload, col_settings = st.columns([3, 1], gap="large")

with col_upload:
    card_head(
        "ico-cyan", "📤", "Upload Lab Report(s)",
        "Drag & drop or browse — JPG, JPEG, PNG or PDF · single or batch",
    )
    uploaded_files = st.file_uploader(
        "Upload Lab Report Images or PDFs",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

with col_settings:
    card_head("ico-purple", "⚙️", "Detection Config", "Fine-tune the neural pipeline")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=float(cfg.model.default_confidence),
        step=0.05,
        help="Minimum confidence score for a detection to be shown.",
    )

    show_labels = st.toggle("Show Labels", value=True)
    show_conf = st.toggle("Show Confidence", value=True)

    # ── OCR engine selection (availability-aware) ──
    engine_options = ocr_mod.available_engines()
    if not engine_options:
        engine_options = ["None Available"]

    default_engine = cfg.ocr.default_engine
    default_engine_idx = (
        engine_options.index(default_engine)
        if default_engine in engine_options else 0
    )
    ocr_engine = st.selectbox(
        "OCR Engine",
        options=engine_options,
        index=default_engine_idx,
        help="Auto tries Tesseract first and falls back to EasyOCR if "
             "Tesseract fails or finds no text.",
    )

# ── Preprocessing controls (defaults sourced from config) ──
pp_cfg = cfg.preprocess
with st.expander("🎛️  Image Preprocessing", expanded=False):
    st.markdown(
        '<div class="card-sub">Optional cleanup applied to each crop before OCR. '
        'Helps on noisy scans and small text.</div>',
        unsafe_allow_html=True,
    )
    pp_enabled = st.toggle(
        "Enable Preprocessing", value=bool(getattr(pp_cfg, "enabled", False))
    )
    pc1, pc2 = st.columns(2)
    with pc1:
        pp_grayscale = st.toggle(
            "Grayscale", value=bool(getattr(pp_cfg, "grayscale", True)),
            disabled=not pp_enabled,
        )
        pp_denoise = st.toggle(
            "Denoise", value=bool(getattr(pp_cfg, "denoise", True)),
            disabled=not pp_enabled,
        )
    with pc2:
        pp_threshold = st.toggle(
            "Adaptive Threshold", value=bool(getattr(pp_cfg, "adaptive_threshold", False)),
            disabled=not pp_enabled,
        )
        pp_sharpen = st.toggle(
            "Sharpen", value=bool(getattr(pp_cfg, "sharpen", False)),
            disabled=not pp_enabled,
        )
    pp_resize = st.number_input(
        "Upscale shortest side to (px, 0 = off)",
        min_value=0, max_value=4000,
        value=int(getattr(pp_cfg, "resize_min_dim", 1000)),
        step=100, disabled=not pp_enabled,
        help="Small crops are upscaled before OCR for better accuracy.",
    )

preprocess_options = {
    "enabled": pp_enabled,
    "grayscale": pp_grayscale,
    "denoise": pp_denoise,
    "adaptive_threshold": pp_threshold,
    "sharpen": pp_sharpen,
    "resize_min_dim": int(pp_resize),
}

# ── Action buttons: Run + Reset ──
act1, act2, _ = st.columns([1.4, 1, 3])
with act1:
    run_clicked = st.button("🚀  Run Detection", use_container_width=True)
with act2:
    reset_clicked = st.button("♻️  Reset", use_container_width=True)

if reset_clicked:
    for key in ("bundles", "rows_map", "rows_order", "run_meta", "run_dir"):
        st.session_state.pop(key, None)
    logger.info("Session reset by user")
    st.rerun()

neon()


# ----------------------------------------------------------
# Build the list of input pages from the uploaded files
# ----------------------------------------------------------

def build_inputs(files, max_files):
    """Turn uploaded files into a list of (source, page, image_rgb).

    Images become a single page; PDFs expand to one entry per page.
    Honours the ``max_files`` safety cap and reports what was dropped.
    """
    inputs = []
    skipped = 0
    for f in files:
        name = f.name
        is_pdf = name.lower().endswith(".pdf") or f.type == "application/pdf"
        try:
            if is_pdf:
                if not pdf_mod.PYMUPDF_AVAILABLE:
                    pill("err", f"✕ &nbsp;Cannot read <code>{name}</code> — PyMuPDF not installed")
                    continue
                remaining = max_files - len(inputs)
                if remaining <= 0:
                    skipped += 1
                    continue
                pages = pdf_mod.pdf_bytes_to_images(
                    f.getvalue(), dpi=cfg.pdf.render_dpi, max_pages=remaining,
                )
                for page_idx, img_rgb in pages:
                    inputs.append((name, page_idx, img_rgb))
            else:
                if len(inputs) >= max_files:
                    skipped += 1
                    continue
                img = Image.open(f).convert("RGB")
                inputs.append((name, 1, np.array(img)))
        except Exception as exc:
            pill("err", f"✕ &nbsp;Failed to read <code>{name}</code>: {exc}")
            logger.exception("Failed to read %s", name)
    return inputs, skipped


# ----------------------------------------------------------
# Run the pipeline (single, batch, or PDF) when requested
# ----------------------------------------------------------

if run_clicked:
    if not uploaded_files:
        pill("info", "🔍 &nbsp;Upload at least one image or PDF first")
    elif ocr_engine == "None Available":
        pill("err", "✕ &nbsp;No OCR engine available — install pytesseract or easyocr")
    else:
        max_files = int(getattr(cfg.batch, "max_files", 25))
        inputs, skipped = build_inputs(uploaded_files, max_files)

        if not inputs:
            pill("err", "✕ &nbsp;Nothing to process — no readable pages found")
        else:
            if skipped:
                pill("info", f"⚠️ &nbsp;Batch cap reached — {skipped} extra file(s)/page(s) skipped "
                             f"(limit {max_files})")

            # Build EasyOCR reader only if the selected engine needs it.
            easyocr_reader = None
            if ocr_engine in (ocr_mod.ENGINE_EASYOCR, ocr_mod.ENGINE_AUTO) and ocr_mod.EASYOCR_AVAILABLE:
                try:
                    easyocr_reader = get_easyocr_reader(
                        tuple(getattr(cfg.ocr, "easyocr_languages", ["en"])),
                        bool(getattr(cfg.ocr, "easyocr_gpu", False)),
                    )
                except Exception as exc:
                    logger.warning("EasyOCR reader unavailable: %s", exc)

            run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{run_stamp}"
            run_dir = os.path.join(cfg.paths.outputs_dir, run_id)

            bundles = []
            progress = st.progress(0.0, text="⚡ Neural network analysing pages…")
            with st.spinner("⚡ Detecting fields and extracting text…"):
                for i, (source, page, image_rgb) in enumerate(inputs, start=1):
                    bundle = process_one(
                        image_rgb, model,
                        source=source, page=page,
                        confidence=confidence_threshold,
                        ocr_engine=ocr_engine,
                        easyocr_reader=easyocr_reader,
                        tesseract_psm=int(getattr(cfg.ocr, "tesseract_psm", 6)),
                        preprocess_options=preprocess_options,
                        show_labels=show_labels,
                        show_conf=show_conf,
                    )
                    bundles.append(bundle)
                    # Structured per-page log record.
                    logging_mod.log_run_record(
                        cfg.paths.logs_dir,
                        bundle_log_record(bundle, run_id, ocr_engine),
                    )
                    if bundle.get("error"):
                        logger.error("Page %s/%s failed: %s", source, page, bundle["error"])
                    progress.progress(i / len(inputs),
                                      text=f"⚡ Processed {i}/{len(inputs)} page(s)")
            progress.empty()

            # ── Flatten report rows into a uid-keyed editable store ──
            rows_map, rows_order = {}, []
            for b in bundles:
                for r in b.get("report_rows", []):
                    rows_map[r["uid"]] = dict(r)
                    rows_order.append(r["uid"])

            # ── Persist outputs to disk (best-effort) ──
            save_error = None
            try:
                os.makedirs(os.path.join(run_dir, "crops"), exist_ok=True)
                for b in bundles:
                    tag = f"{os.path.splitext(b['source'])[0]}_p{b['page']}"
                    Image.fromarray(np.asarray(b["annotated_rgb"])).save(
                        os.path.join(run_dir, f"annotated_{tag}.png")
                    )
                    for d in b.get("detections", []):
                        crop = d["crop"]
                        if getattr(crop, "size", 0) > 0:
                            safe = d["class"].replace(" ", "_")
                            Image.fromarray(np.asarray(crop)).save(
                                os.path.join(run_dir, "crops", f"{tag}_{d['index']}_{safe}.png")
                            )
                records = exports_mod.rows_to_records(list(rows_map.values()))
                summary = summarise_fields(list(rows_map.values()))
                with open(os.path.join(run_dir, "ocr_results.csv"), "wb") as fh:
                    fh.write(exports_mod.records_to_csv_bytes(records))
                with open(os.path.join(run_dir, "ocr_results.json"), "wb") as fh:
                    fh.write(exports_mod.records_to_json_bytes(records, summary,
                             {"run_id": run_id, "engine": ocr_engine}))
                if exports_mod.excel_available():
                    with open(os.path.join(run_dir, "ocr_results.xlsx"), "wb") as fh:
                        fh.write(exports_mod.records_to_excel_bytes(records, summary))
            except Exception as exc:
                save_error = str(exc)
                logger.exception("Failed to persist outputs")

            # ── Store everything in session for interactive rendering ──
            st.session_state["bundles"] = bundles
            st.session_state["rows_map"] = rows_map
            st.session_state["rows_order"] = rows_order
            st.session_state["run_dir"] = run_dir
            st.session_state["run_meta"] = {
                "run_id": run_id,
                "timestamp": run_stamp,
                "engine": ocr_engine,
                "confidence": confidence_threshold,
                "num_files": len({b["source"] for b in bundles}),
                "num_pages": len(bundles),
                "model": minfo.get("name", "—"),
                "preprocess": preprocess_options,
                "save_error": save_error,
                "upload_sig": [(f.name, f.size) for f in uploaded_files],
            }
            logger.info("Run %s complete: %d page(s), %d detection(s)",
                        run_id, len(bundles), sum(b["num_detections"] for b in bundles))


# ----------------------------------------------------------
# Render results (reads from session_state so search / edits
# survive Streamlit re-runs without recomputing detection)
# ----------------------------------------------------------

bundles = st.session_state.get("bundles")

# Invalidate stale results if the uploaded file set changed since the run.
if bundles:
    current_sig = [(f.name, f.size) for f in (uploaded_files or [])]
    if current_sig != st.session_state.get("run_meta", {}).get("upload_sig"):
        bundles = None

if bundles:
    meta = st.session_state["run_meta"]
    rows_map = st.session_state["rows_map"]
    rows_order = st.session_state["rows_order"]
    run_dir = st.session_state["run_dir"]

    # ── Aggregate metrics across all pages ──
    total_det = sum(b["num_detections"] for b in bundles)
    all_classes = set()
    for b in bundles:
        all_classes.update(b.get("classes", {}).keys())
    confs = [r["Confidence"] for r in rows_map.values()
             if isinstance(r.get("Confidence"), (int, float))]
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0
    total_time = sum(b["total_time"] for b in bundles)
    ocr_ok = sum(b["ocr_success_count"] for b in bundles)

    card_head("ico-pink", "📈", "Detection Results",
              f"{meta['num_pages']} page(s) from {meta['num_files']} file(s) · engine: {meta['engine']}")

    stats_row([
        (total_det, "Detections"),
        (len(all_classes), "Classes Found"),
        (f"{avg_conf:.1f}%", "Avg Confidence"),
        (f"{total_time:.2f}s", "Total Time"),
    ])
    stats_row([
        (f"{ocr_ok}/{total_det}" if total_det else "0/0", "OCR Success"),
        (meta["num_pages"], "Pages"),
        (f"{meta['confidence']:.0%}", "Threshold"),
        ("ON" if meta["preprocess"]["enabled"] else "OFF", "Preprocess"),
    ])

    if meta.get("save_error"):
        pill("err", f"✕ &nbsp;Outputs not fully saved: {meta['save_error']}")
    else:
        pill("ok", f"<span class='live-dot'></span> &nbsp;Outputs saved to <code>{run_dir}/</code>")

    neon()

    # ── Per-page annotated image + crop gallery ──
    card_head("ico-cyan", "🖼️", "Annotated Pages", "Detected regions overlaid on each page")
    for b in bundles:
        header = f"{b['source']} — page {b['page']}"
        with st.expander(header, expanded=(len(bundles) == 1)):
            if b.get("error"):
                pill("err", f"✕ &nbsp;{b['error']}")
                continue
            st.markdown('<div class="scan-wrapper">', unsafe_allow_html=True)
            st.image(b["annotated_rgb"], caption=f"🔍 {header}", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if b["num_detections"] == 0:
                pill("info", "🔍 &nbsp;No detections — try lowering the confidence threshold")
                continue

            dets = b["detections"]
            crop_cols = st.columns(min(len(dets), 4))
            for idx, d in enumerate(dets):
                col = crop_cols[idx % len(crop_cols)]
                with col:
                    if getattr(d["crop"], "size", 0) > 0:
                        st.image(
                            d["crop"],
                            caption=f"{d['class']} ({d['confidence']}%)",
                            use_column_width=True,
                        )

    if total_det == 0:
        pill("info", "🔍 &nbsp;No detections across any page")
    else:
        neon()

        # ── Editable structured report table + search / filter ──
        card_head("ico-purple", "🔤", "Structured Report (Editable)",
                  "Review, search and correct extracted text — edits flow into every export")

        search = st.text_input(
            "🔎 Search fields / text",
            value="",
            placeholder="Filter by field, class or extracted text…",
            label_visibility="collapsed",
        )

        display_cols = ["Source", "Page", "Order", "Field", "Detected Class",
                        "Text", "Confidence", "OCR Engine"]

        def _matches(row, q):
            if not q:
                return True
            q = q.lower()
            return any(q in str(row.get(c, "")).lower()
                       for c in ("Field", "Detected Class", "Text"))

        visible_uids = [uid for uid in rows_order if _matches(rows_map[uid], search)]
        table_data = []
        for uid in visible_uids:
            r = rows_map[uid]
            entry = {"uid": uid}
            entry.update({c: r.get(c, "") for c in display_cols})
            table_data.append(entry)

        if not table_data:
            pill("info", "🔍 &nbsp;No rows match your search")
        else:
            df = pd.DataFrame(table_data)
            edited = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                key=f"editor_{meta['run_id']}_{search}",
                column_config={
                    "uid": None,  # hidden helper key
                    "Source": st.column_config.TextColumn("Source", disabled=True),
                    "Page": st.column_config.NumberColumn("Page", disabled=True),
                    "Order": st.column_config.NumberColumn("#", disabled=True),
                    "Field": st.column_config.TextColumn("Field", help="Editable"),
                    "Detected Class": st.column_config.TextColumn("Class", disabled=True),
                    "Text": st.column_config.TextColumn("Extracted Text", help="Editable"),
                    "Confidence": st.column_config.NumberColumn("Conf %", disabled=True, format="%.1f"),
                    "OCR Engine": st.column_config.TextColumn("Engine", disabled=True),
                },
            )
            # Write edits back into the master store (only for editable cols).
            for _, row in edited.iterrows():
                uid = row["uid"]
                if uid in rows_map:
                    rows_map[uid]["Field"] = row["Field"]
                    rows_map[uid]["Text"] = row["Text"]
            st.session_state["rows_map"] = rows_map

            pill("info", f"📊 &nbsp;Showing {len(table_data)} of {len(rows_order)} row(s)")

        # ── Best-effort key/value summary of singleton fields ──
        summary = summarise_fields(list(rows_map.values()))
        if summary:
            neon()
            card_head("ico-green", "🧾", "Report Summary",
                      "Highest-confidence value for each singleton field")
            summ_rows = "".join(
                f'<div class="info-row"><span class="info-key">{k}</span>'
                f'<span class="info-val">{str(v)}</span></div>'
                for k, v in summary.items()
            )
            st.markdown(
                f'<div class="holo-card"><div class="info-grid">{summ_rows}</div></div>',
                unsafe_allow_html=True,
            )

        neon()

        # ── Download / export section (uses the possibly-edited rows) ──
        card_head("ico-green", "💾", "Download Results",
                  "Export edited data as CSV, JSON or Excel, or grab everything as a ZIP")

        ordered_rows = [rows_map[uid] for uid in rows_order]
        records = exports_mod.rows_to_records(ordered_rows)
        export_summary = summarise_fields(ordered_rows)
        export_meta = {
            "run_id": meta["run_id"],
            "timestamp": meta["timestamp"],
            "engine": meta["engine"],
            "model": meta["model"],
            "confidence": meta["confidence"],
            "pages": meta["num_pages"],
            "files": meta["num_files"],
        }

        csv_bytes = exports_mod.records_to_csv_bytes(records)
        json_bytes = exports_mod.records_to_json_bytes(records, export_summary, export_meta)

        # Assemble image assets for the ZIP.
        zip_images, zip_crops = [], []
        for b in bundles:
            tag = f"{os.path.splitext(b['source'])[0]}_p{b['page']}"
            try:
                zip_images.append((f"annotated_{tag}.png", png_bytes(b["annotated_rgb"])))
            except Exception:
                pass
            for d in b.get("detections", []):
                if getattr(d["crop"], "size", 0) > 0:
                    safe = d["class"].replace(" ", "_")
                    try:
                        zip_crops.append((f"{tag}_{d['index']}_{safe}.png", png_bytes(d["crop"])))
                    except Exception:
                        pass

        zip_bytes = exports_mod.build_zip(
            records, summary=export_summary, meta=export_meta,
            images=zip_images, crops=zip_crops,
        )

        stamp = meta["timestamp"]
        dl1, dl2, dl3, dl4 = st.columns(4)
        with dl1:
            st.download_button("📄  CSV", data=csv_bytes,
                               file_name=f"ocr_results_{stamp}.csv",
                               mime="text/csv", use_container_width=True)
        with dl2:
            st.download_button("📋  JSON", data=json_bytes,
                               file_name=f"ocr_results_{stamp}.json",
                               mime="application/json", use_container_width=True)
        with dl3:
            if exports_mod.excel_available():
                try:
                    xlsx_bytes = exports_mod.records_to_excel_bytes(records, export_summary)
                    st.download_button(
                        "📊  Excel", data=xlsx_bytes,
                        file_name=f"ocr_results_{stamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.button("📊  Excel N/A", disabled=True, use_container_width=True,
                              help=f"Excel export unavailable: {exc}")
            else:
                st.button("📊  Excel N/A", disabled=True, use_container_width=True,
                          help="Install openpyxl to enable Excel export")
        with dl4:
            st.download_button("📦  ZIP (All)", data=zip_bytes,
                               file_name=f"lab_report_ocr_{stamp}.zip",
                               mime="application/zip", use_container_width=True)

elif uploaded_files:
    # Files staged but not yet processed — show a lightweight preview.
    card_head("ico-green", "🖼️", "Ready to Process",
              f"{len(uploaded_files)} file(s) staged — click Run Detection to begin")
    prev_cols = st.columns(min(len(uploaded_files), 4))
    for i, f in enumerate(uploaded_files):
        with prev_cols[i % len(prev_cols)]:
            is_pdf = f.name.lower().endswith(".pdf") or f.type == "application/pdf"
            if is_pdf:
                st.markdown(
                    f'<div class="holo-card" style="text-align:center;padding:2rem 1rem;">'
                    f'<div style="font-size:2.5rem;">📄</div>'
                    f'<div class="card-sub">{f.name}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                try:
                    st.image(Image.open(f), caption=f.name, use_column_width=True)
                except Exception:
                    st.markdown(f'<div class="card-sub">{f.name}</div>', unsafe_allow_html=True)

else:
    # ── Empty State ──
    st.markdown("""
    <div class="holo-card">
        <div class="empty-state">
            <div class="empty-icon">🔬</div>
            <div class="empty-title">No Image Uploaded</div>
            <p class="empty-desc">
                Upload a lab report image or PDF above to begin AI-powered
                detection and OCR analysis.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------

st.markdown("""
<div class="futr-footer">
    <div class="footer-text">Lab Report OCR &nbsp;◆&nbsp; Powered by YOLO &nbsp;◆&nbsp; Streamlit</div>
    <div class="footer-brand">Neural Vision Engine v2.0</div>
</div>
""", unsafe_allow_html=True)
