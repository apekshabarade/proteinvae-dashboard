"""
ProteinVAE – AI-Driven De Novo Protein Generation
Professional biotech research dashboard (Streamlit).
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from model_loader import (
    generate_protein_sequence,
    get_device,
    load_config,
    load_model,
    load_research_metrics,
    save_generated_to_files,
)
from utils import (
    analyze_sequence,
    compute_pca_2d,
    compute_tsne_2d,
    export_csv_download,
    export_fasta_download,
    generate_latent_space_data,
    load_generated_proteins,
    plot_aa_frequency,
    plot_architecture_flow,
    plot_gauge,
    plot_latent_scatter,
    plot_sequence_profile,
    load_fasta_content,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ProteinVAE | AI Protein Generation",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent

# ── High-contrast neon biotech theme CSS ───────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;800&family=Inter:wght@500;600;700&display=swap');

:root {
    --neon-cyan: #00F5FF;
    --neon-purple: #8B5CF6;
    --neon-blue: #2563EB;
    --neon-green: #00FF9D;
    --text-white: #FFFFFF;
    --text-secondary: #D1D5DB;
    --bg-dark: #050816;
    --card-bg: rgba(28, 40, 68, 0.98);
    --sidebar-bg: #010308;
}

/* ── Base & background ───────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-white) !important;
}

.stApp {
    background:
        radial-gradient(ellipse 70% 45% at 15% 8%, rgba(37, 99, 235, 0.22) 0%, transparent 50%),
        radial-gradient(ellipse 55% 35% at 90% 85%, rgba(139, 92, 246, 0.18) 0%, transparent 45%),
        linear-gradient(165deg, #050816 0%, #0c1428 50%, #050816 100%) !important;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 1.5rem;
    max-width: 1200px;
}

/* ── Global text force (main content) ─────────────────────────────────────── */
.main p, .main span, .main li, .main label,
.main [data-testid="stMarkdownContainer"] p,
.main [data-testid="stMarkdownContainer"] li,
.main [data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdown"] p,
[data-testid="stMarkdown"] li {
    color: var(--text-white) !important;
    opacity: 1 !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    line-height: 1.75 !important;
}

label[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] p {
    color: var(--text-white) !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    opacity: 1 !important;
}

/* ── Typography / headings ──────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6,
.main h1, .main h2, .main h3, .main h4,
[data-testid="stMarkdown"] h1,
[data-testid="stMarkdown"] h2,
[data-testid="stMarkdown"] h3,
[data-testid="stMarkdown"] h4,
.body-text h4 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-cyan) !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    text-shadow: 0 0 18px rgba(0, 245, 255, 0.45) !important;
}

[data-testid="stMarkdown"] strong, .body-text strong {
    color: var(--text-white) !important;
    font-weight: 700 !important;
}

.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(2rem, 5vw, 3rem);
    font-weight: 800;
    color: var(--neon-cyan) !important;
    -webkit-text-fill-color: var(--neon-cyan) !important;
    margin-bottom: 0.25rem;
    letter-spacing: 0.06em;
    text-shadow: 0 0 24px rgba(0, 245, 255, 0.7);
}

.hero-subtitle {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.3rem;
    color: var(--neon-cyan) !important;
    font-weight: 700;
    margin-bottom: 1rem;
    text-shadow: 0 0 14px rgba(0, 245, 255, 0.5);
}

.body-text {
    color: var(--text-white) !important;
    font-size: 1.08rem;
    line-height: 1.85;
    font-weight: 600;
}

.section-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.4rem;
    color: var(--neon-cyan) !important;
    font-weight: 700;
    border-left: 4px solid var(--neon-cyan);
    padding: 0.55rem 0 0.55rem 14px;
    margin: 1.5rem 0 1rem 0;
    background: rgba(0, 245, 255, 0.1);
    border-radius: 0 8px 8px 0;
    text-shadow: 0 0 14px rgba(0, 245, 255, 0.5);
}

/* ── Stat / metric cards ──────────────────────────────────────────────────── */
.stat-card {
    background: var(--card-bg);
    border: 2px solid rgba(0, 245, 255, 0.5);
    border-radius: 16px;
    padding: 1.5rem 1.2rem;
    text-align: center;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.4), 0 0 24px rgba(0, 245, 255, 0.15);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45), 0 0 32px rgba(0, 245, 255, 0.3);
}

.stat-value {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.7rem, 3vw, 2.3rem);
    color: var(--neon-cyan);
    font-weight: 800;
    text-shadow: 0 0 22px rgba(0, 245, 255, 0.65);
}

.stat-label {
    color: var(--text-white);
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    opacity: 1;
}

/* ── Custom metric grid (Protein Generation & analysis) ─────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(180px, 1fr));
    gap: 1.15rem;
    width: 100%;
    margin: 1.25rem 0 1rem;
    align-items: stretch;
}

.metric-card {
    min-width: 180px;
    width: 100%;
    background: rgba(20, 32, 58, 0.82);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 2px solid rgba(0, 245, 255, 0.55);
    border-radius: 16px;
    padding: 1.35rem 1.2rem;
    box-shadow:
        0 10px 36px rgba(0, 0, 0, 0.45),
        0 0 28px rgba(0, 245, 255, 0.18),
        inset 0 1px 0 rgba(255, 255, 255, 0.08);
    overflow: visible !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: stretch;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 245, 255, 0.85);
    box-shadow:
        0 14px 44px rgba(0, 0, 0, 0.5),
        0 0 36px rgba(0, 245, 255, 0.35);
}

.metric-label {
    color: #FFFFFF !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    line-height: 1.35 !important;
    letter-spacing: 0.02em;
    margin-bottom: 0.65rem !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-wrap: break-word !important;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.35);
}

.metric-value {
    font-family: 'Orbitron', sans-serif !important;
    color: #00F5FF !important;
    font-size: clamp(1.5rem, 3.5vw, 2.8rem) !important;
    font-weight: 800 !important;
    opacity: 1 !important;
    line-height: 1.15 !important;
    text-shadow: 0 0 12px rgba(0, 245, 255, 0.8) !important;
    overflow: visible !important;
    white-space: nowrap !important;
}

.metric-grid--3 {
    grid-template-columns: repeat(3, minmax(180px, 1fr));
}

@media (max-width: 1100px) {
    .metric-grid,
    .metric-grid--3 {
        grid-template-columns: repeat(2, minmax(180px, 1fr));
    }
}

@media (max-width: 520px) {
    .metric-grid {
        grid-template-columns: 1fr;
    }
    .metric-card {
        min-width: 100%;
    }
    .metric-value {
        font-size: 2.2rem !important;
    }
}

/* ── Custom progress bars (Metrics page) ──────────────────────────────────── */
.progress-row {
    margin: 1.35rem 0 1.65rem;
}
.progress-label {
    color: #FFFFFF !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.65rem !important;
    letter-spacing: 0.03em;
    text-shadow: 0 0 14px rgba(0, 245, 255, 0.55);
    opacity: 1 !important;
}
.progress-track {
    height: 20px;
    background: rgba(35, 50, 80, 0.95);
    border: 2px solid rgba(0, 245, 255, 0.45);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.35);
}
.progress-fill {
    height: 100%;
    min-width: 4px;
    background: linear-gradient(90deg, #00F5FF 0%, #8B5CF6 50%, #00FF9D 100%);
    border-radius: 10px;
    box-shadow: 0 0 22px rgba(0, 245, 255, 0.75);
}

/* ── Architecture table ───────────────────────────────────────────────────── */
.arch-table-wrap {
    background: rgba(28, 40, 68, 0.98);
    border: 1px solid rgba(0, 245, 255, 0.45);
    border-radius: 14px;
    padding: 0.25rem;
    margin: 0.5rem 0 1rem;
    box-shadow:
        0 0 24px rgba(0, 245, 255, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
    overflow: hidden;
    animation: fadeInContent 0.7s ease-out;
}

.arch-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 1rem;
}

.arch-table th {
    font-family: 'Orbitron', sans-serif;
    color: var(--neon-cyan);
    background: rgba(37, 99, 235, 0.35);
    padding: 14px 16px;
    text-align: left;
    font-weight: 700;
    border-bottom: 2px solid rgba(0, 245, 255, 0.5);
    text-shadow: 0 0 10px rgba(0, 245, 255, 0.3);
}

.arch-table td {
    color: #FFFFFF;
    padding: 12px 16px;
    font-weight: 600;
    border-bottom: 1px solid rgba(0, 245, 255, 0.2);
}

.arch-table tr:nth-child(even) td {
    background: rgba(0, 245, 255, 0.06);
}

.arch-table tr:nth-child(odd) td {
    background: rgba(139, 92, 246, 0.05);
}

.arch-table tr:hover td {
    background: rgba(0, 245, 255, 0.12);
    color: #FFFFFF;
}

/* Generic markdown tables */
[data-testid="stMarkdown"] table {
    width: 100%;
    background: rgba(20, 30, 55, 0.9) !important;
    border: 1px solid rgba(0, 245, 255, 0.4) !important;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.15);
}

[data-testid="stMarkdown"] table th {
    background: rgba(37, 99, 235, 0.4) !important;
    color: var(--neon-cyan) !important;
    font-family: 'Orbitron', sans-serif;
    padding: 12px !important;
}

[data-testid="stMarkdown"] table td {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    padding: 10px 12px !important;
    border-color: rgba(0, 245, 255, 0.2) !important;
}

[data-testid="stMarkdown"] table tr:nth-child(even) {
    background: rgba(0, 245, 255, 0.05) !important;
}

/* ── Sequence box & pills ─────────────────────────────────────────────────── */
.seq-box {
    background: rgba(22, 35, 58, 0.98);
    border: 1px solid rgba(0, 255, 157, 0.5);
    border-radius: 12px;
    padding: 1.25rem 1.1rem;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.95rem;
    color: var(--neon-green);
    word-break: break-all;
    line-height: 1.7;
    box-shadow:
        0 0 24px rgba(0, 255, 157, 0.15),
        inset 0 0 40px rgba(0, 255, 157, 0.03);
    text-shadow: 0 0 8px rgba(0, 255, 157, 0.25);
}

.metric-pill {
    display: inline-block;
    background: rgba(139, 92, 246, 0.25);
    border: 1px solid rgba(139, 92, 246, 0.7);
    border-radius: 24px;
    padding: 6px 16px;
    margin: 5px;
    color: #FFFFFF;
    font-size: 0.88rem;
    font-weight: 500;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.metric-pill:hover {
    transform: scale(1.05);
    box-shadow: 0 0 16px rgba(139, 92, 246, 0.5);
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #010308 0%, #020510 100%) !important;
    border-right: 2px solid rgba(0, 245, 255, 0.55) !important;
    box-shadow: 4px 0 40px rgba(0, 245, 255, 0.12) !important;
}

section[data-testid="stSidebar"] > div {
    padding: 1.75rem 1rem 1.5rem !important;
}

/* Force ALL sidebar text bright */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
section[data-testid="stSidebar"] [data-testid="stRadio"] label span,
section[data-testid="stSidebar"] [data-testid="stRadio"] label div {
    color: #FFFFFF !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] h1 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-cyan) !important;
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    text-shadow: 0 0 22px rgba(0, 245, 255, 0.6) !important;
}

section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: var(--text-secondary) !important;
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(0, 245, 255, 0.35) !important;
    margin: 1.4rem 0 !important;
}

.sidebar-info p, .sidebar-info li {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.98rem !important;
    line-height: 1.7 !important;
}

.sidebar-info strong {
    color: var(--neon-cyan) !important;
}

.sidebar-info code {
    color: var(--neon-green) !important;
    background: rgba(0, 245, 255, 0.12) !important;
    padding: 2px 6px;
    border-radius: 4px;
}

section[data-testid="stSidebar"] [data-testid="stAlert"] p {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Sidebar radio navigation */
section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 0.85rem !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(30, 45, 75, 0.95) !important;
    border: 2px solid rgba(0, 245, 255, 0.35) !important;
    border-radius: 12px !important;
    padding: 0.95rem 1.15rem !important;
    margin-bottom: 6px !important;
    transition: all 0.25s ease !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    border-color: var(--neon-cyan) !important;
    background: rgba(37, 99, 235, 0.45) !important;
    box-shadow: 0 0 22px rgba(0, 245, 255, 0.35) !important;
    transform: translateX(6px);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
    background: linear-gradient(90deg, rgba(0, 245, 255, 0.25), rgba(37, 99, 235, 0.35)) !important;
    border: 2px solid var(--neon-cyan) !important;
    box-shadow: 0 0 28px rgba(0, 245, 255, 0.5), inset 0 0 16px rgba(0, 245, 255, 0.12) !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    text-shadow: 0 0 12px rgba(0, 245, 255, 0.6);
}

/* ── Streamlit native metrics (Download / fallback) ───────────────────────── */
[data-testid="stMetric"] {
    background: rgba(20, 32, 58, 0.82) !important;
    border: 2px solid rgba(0, 245, 255, 0.55) !important;
    border-radius: 16px;
    padding: 1.25rem 1.2rem !important;
    min-width: 180px !important;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.4), 0 0 24px rgba(0, 245, 255, 0.15);
    overflow: visible !important;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] label,
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricLabel"] div {
    color: #FFFFFF !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.35);
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    font-family: 'Orbitron', sans-serif !important;
    color: #00F5FF !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    text-shadow: 0 0 12px rgba(0, 245, 255, 0.8) !important;
    opacity: 1 !important;
    overflow: visible !important;
}

div[data-testid="stMetricDelta"] {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, #00F5FF 0%, #2563EB 50%, #8B5CF6 100%) !important;
    color: #050816 !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    letter-spacing: 0.04em;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(0, 245, 255, 0.3);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    box-shadow:
        0 0 28px rgba(0, 245, 255, 0.55),
        0 0 40px rgba(139, 92, 246, 0.35) !important;
    transform: scale(1.04) translateY(-2px) !important;
    color: #030610 !important;
}

.stButton > button:active,
.stDownloadButton > button:active {
    transform: scale(0.98) !important;
}

/* Primary generate button emphasis */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #00FF9D 0%, #00F5FF 50%, #8B5CF6 100%) !important;
    box-shadow: 0 0 24px rgba(0, 255, 157, 0.35);
}

/* ── Inputs: select, slider, text ─────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1px solid rgba(0, 245, 255, 0.4) !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
    font-size: 1rem !important;
}

[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 16px rgba(0, 245, 255, 0.35) !important;
}

[data-testid="stSlider"] label,
[data-testid="stSlider"] label p,
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] label p,
[data-testid="stTextArea"] label,
[data-testid="stTextArea"] label p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    opacity: 1 !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] div {
    background: rgba(0, 245, 255, 0.25) !important;
}

[data-testid="stSlider"] [role="slider"] {
    background: var(--neon-cyan) !important;
    box-shadow: 0 0 12px rgba(0, 245, 255, 0.6) !important;
}

/* ── Alerts & info boxes ──────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-width: 2px !important;
    opacity: 1 !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    opacity: 1 !important;
}

div[data-baseweb="notification"] {
    background: rgba(28, 40, 68, 0.98) !important;
}

/* ── Code blocks ──────────────────────────────────────────────────────────── */
[data-testid="stCode"] pre,
.stCode pre {
    background: rgba(10, 18, 35, 0.95) !important;
    border: 1px solid rgba(0, 245, 255, 0.35) !important;
    border-radius: 12px !important;
    color: var(--neon-green) !important;
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.1);
}

/* ── Dataframe ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0, 245, 255, 0.35);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.12);
}

/* ── Native Streamlit progress (fallback) ─────────────────────────────────── */
[data-testid="stProgress"] label,
[data-testid="stProgress"] p,
[data-testid="stProgress"] span,
.stProgress label,
.stProgress p {
    color: #FFFFFF !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    text-shadow: 0 0 12px rgba(0, 245, 255, 0.5);
    margin-bottom: 0.5rem !important;
}

[data-testid="stProgress"] > div,
.stProgress > div {
    height: 20px !important;
    min-height: 20px !important;
}

[data-testid="stProgress"] > div > div,
.stProgress > div > div {
    background: rgba(35, 50, 80, 0.95) !important;
    border-radius: 12px !important;
    border: 2px solid rgba(0, 245, 255, 0.35) !important;
    height: 20px !important;
}

[data-testid="stProgress"] > div > div > div,
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00F5FF, #8B5CF6, #00FF9D) !important;
    border-radius: 10px !important;
    box-shadow: 0 0 18px rgba(0, 245, 255, 0.65) !important;
    height: 100% !important;
}

/* ── Plotly charts container ──────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(0, 245, 255, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

/* ── Streamlit baseweb / theme text overrides ─────────────────────────────── */
label[data-baseweb="label"],
label[data-baseweb="label"] p,
label[data-baseweb="label"] div {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] label[data-baseweb="label"],
section[data-testid="stSidebar"] [data-testid="stRadio"] label * {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* ── Hide Streamlit chrome noise ──────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: transparent;
}

/* ── Mobile responsive ────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .hero-title { font-size: 1.85rem; }
    .stat-card { padding: 1.1rem 0.8rem; }
    .stat-value { font-size: 1.5rem; }
    .section-header { font-size: 1.15rem; }
    .main .block-container { padding-left: 1rem; padding-right: 1rem; }
    .arch-table { font-size: 0.88rem; }
    .arch-table th, .arch-table td { padding: 10px 12px; }
}

/* ── Latest generated protein highlight ───────────────────────────────────── */
.latest-highlight {
    background: rgba(0, 245, 255, 0.1);
    border: 2px solid rgba(0, 245, 255, 0.55);
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 0 24px rgba(0, 245, 255, 0.2);
}

.latest-highlight .protein-meta-title {
    font-family: 'Orbitron', sans-serif;
    color: #00F5FF;
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 0.35rem;
}

.latest-highlight .protein-meta-detail {
    color: #D1D5DB;
    font-size: 0.95rem;
    font-weight: 600;
    line-height: 1.6;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading ProteinVAE model…")
def _cached_model():
    return load_model()


LATEST_PROTEIN_KEY = "__latest__"
CUSTOM_INPUT_KEY = "__custom__"


def init_session_storage() -> None:
    """Initialize global session keys for generated proteins."""
    defaults: dict = {
        "generated_protein": None,
        "generated_history": [],
        "last_sequence": "",
        "last_type": "",
        "last_id": "",
        "force_analysis_latest": False,
        "analysis_select_key": LATEST_PROTEIN_KEY,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def store_generated_protein(sequence: str, protein_type: str, protein_id: str) -> None:
    """Persist latest generation and append to session history."""
    stats = analyze_sequence(sequence)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = {
        "sequence": sequence,
        "type": protein_type,
        "id": protein_id,
        "timestamp": timestamp,
        "entropy": stats["entropy"],
        "hydrophobicity": stats["hydrophobicity"],
        "length": stats["length"],
        "dominant_aa": stats["dominant_aa"],
        "confidence": stats["confidence"],
    }

    st.session_state.generated_protein = entry
    st.session_state.generated_history.insert(
        0,
        {
            "sequence": sequence,
            "type": protein_type,
            "id": protein_id,
            "timestamp": timestamp,
            "entropy": stats["entropy"],
            "hydrophobicity": stats["hydrophobicity"],
        },
    )

    # Backward compatibility (Download page, etc.)
    st.session_state.last_sequence = sequence
    st.session_state.last_type = protein_type
    st.session_state.last_id = protein_id
    st.session_state.force_analysis_latest = True
    st.session_state.analysis_select_key = LATEST_PROTEIN_KEY


def build_analysis_sequence_options() -> tuple[list[str], list[str], dict[str, str]]:
    """Return (keys, labels, key→sequence) for the analysis dropdown."""
    keys: list[str] = []
    labels: list[str] = []
    lookup: dict[str, str] = {}

    latest = st.session_state.get("generated_protein")
    if latest and latest.get("sequence"):
        keys.append(LATEST_PROTEIN_KEY)
        labels.append(
            f"⭐ Latest Generated Protein — {latest['id']} ({latest['timestamp']})"
        )
        lookup[LATEST_PROTEIN_KEY] = latest["sequence"]

    seen_ids: set[str] = set()
    if latest:
        seen_ids.add(latest.get("id", ""))

    for item in st.session_state.get("generated_history", []):
        pid = item.get("id", "")
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        key = f"hist_{pid}"
        keys.append(key)
        labels.append(f"{pid} — {item.get('type', 'Unknown')} ({item.get('timestamp', '')})")
        lookup[key] = item["sequence"]

    keys.append(CUSTOM_INPUT_KEY)
    labels.append("Custom Input")
    return keys, labels, lookup


def render_full_sequence_analysis(sequence: str, meta: dict | None = None) -> None:
    """Charts and metrics for a protein sequence."""
    if meta:
        st.markdown(
            f"""
<div class="latest-highlight">
  <div class="protein-meta-title">⭐ Latest Generated Protein</div>
  <div class="protein-meta-detail">
    <strong>ID:</strong> {meta.get("id", "—")} &nbsp;|&nbsp;
    <strong>Type:</strong> {meta.get("type", "—")} &nbsp;|&nbsp;
    <strong>Generated:</strong> {meta.get("timestamp", "—")}
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    stats = analyze_sequence(sequence)
    render_metric_grid([
        ("Length", stats["length"]),
        ("Entropy", f"{stats['entropy']:.3f}"),
        ("Hydrophobicity", f"{stats['hydrophobicity']:.3f}"),
        ("Dominant AA", stats["dominant_aa"]),
    ])
    st.markdown(
        f'<p style="color:#FFFFFF;font-weight:700;font-size:1.05rem;">'
        f'Confidence Score: <span style="color:#00F5FF;">{stats["confidence"]:.1%}</span></p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        " ".join(
            f'<span class="metric-pill">{aa}: {v*100:.1f}%</span>'
            for aa, v in sorted(stats["composition"].items(), key=lambda x: -x[1])[:8]
        ),
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_aa_frequency(sequence), use_container_width=True)
    with col_b:
        st.plotly_chart(plot_sequence_profile(sequence), use_container_width=True)

    st.markdown("**Sequence (formatted)**")
    formatted = " ".join(sequence[i : i + 10] for i in range(0, len(sequence), 10))
    st.markdown(f'<div class="seq-box">{formatted}</div>', unsafe_allow_html=True)


def render_stat_cards():
    cols = st.columns(4)
    stats = [
        ("120K", "Training Sequences"),
        ("128D", "Latent Space"),
        ("100%", "Novelty Score"),
        ("0.9888", "ProtBERT Alignment"),
    ]
    for col, (val, lbl) in zip(cols, stats):
        col.markdown(
            f'<div class="stat-card"><div class="stat-value">{val}</div>'
            f'<div class="stat-label">{lbl}</div></div>',
            unsafe_allow_html=True,
        )


def render_metric_grid(items: list[tuple[str, str | int | float]]) -> None:
    """Full-width custom metric cards — labels never truncate."""
    grid_class = "metric-grid metric-grid--3" if len(items) == 3 else "metric-grid"
    cards = "".join(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>'
        for label, value in items
    )
    st.markdown(f'<div class="{grid_class}">{cards}</div>', unsafe_allow_html=True)


def page_home():
    st.markdown('<p class="hero-title">ProteinVAE</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">AI-Driven De Novo Protein Generation</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="body-text">
<strong>ProteinVAE</strong> is a research-grade <strong>Hybrid Transformer–BiLSTM Variational Autoencoder</strong> trained on
~120,000 protein sequences for biologically plausible <em>de novo</em> sequence design.<br><br>
The model learns a <strong>128-dimensional latent manifold</strong> capturing structural and functional protein
features, enabling conditional generation of enzymes, membrane proteins, and binding proteins with
high <strong>ProtBERT alignment</strong> and measured <strong>sequence novelty</strong>.
</div>
        """,
        unsafe_allow_html=True,
    )
    render_stat_cards()
    st.markdown('<p class="section-header">Architecture Summary</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
<div class="arch-table-wrap">
<table class="arch-table">
<thead><tr><th>Component</th><th>Specification</th></tr></thead>
<tbody>
<tr><td>Encoder</td><td>3-layer Transformer + 2-layer BiLSTM</td></tr>
<tr><td>Latent</td><td>128D Gaussian (μ, log σ²)</td></tr>
<tr><td>Decoder</td><td>3-layer Transformer Decoder</td></tr>
<tr><td>Vocabulary</td><td>24 tokens (20 AA + special)</td></tr>
<tr><td>Embedding</td><td>256D</td></tr>
</tbody>
</table>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        try:
            cfg = load_config()
            st.info(f"**Model:** {cfg.get('model_type', 'ProteinVAE')}")
            st.success(f"**Training:** {cfg.get('training_type', 'Research-Grade VAE')}")
            dev = get_device()
            st.warning(f"**Compute device:** `{dev}`")
        except Exception as e:
            st.error(f"Config load error: {e}")


def page_architecture():
    st.markdown('<p class="section-header">Model Architecture</p>', unsafe_allow_html=True)
    st.plotly_chart(plot_architecture_flow(), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
<div class="body-text">
<h4>Transformer Encoder</h4>
Multi-head self-attention (8 heads) over amino-acid token embeddings
captures long-range sequence dependencies before BiLSTM pooling.
</div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="body-text">
<h4>Variational Latent Space</h4>
The encoder outputs μ and log σ²; reparameterization yields <strong>z ∈ ℝ¹²⁸</strong>.
KL-regularized training ensures smooth, interpolatable latent geometry.
</div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div class="body-text">
<h4>Transformer Decoder</h4>
Cross-attention decoder projects <strong>z</strong> into sequence space with
anti-repetition sampling for biologically valid outputs.
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### VAE Workflow")
    st.code(
        """
Input Sequence → Token Embedding → Transformer Encoder → BiLSTM
    → Fusion Layer → μ, log(σ²) → Reparameterize → z (128D)
    → Latent Projection → Transformer Decoder → AA Logits → Protein Sequence
        """,
        language="text",
    )


def page_generation():
    st.markdown('<p class="section-header">Protein Generation</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        protein_type = st.selectbox("Protein Type", ["Enzyme", "Membrane", "Binding"])
        seq_len = st.slider("Target Sequence Length", 50, 300, 150, step=10)
        temperature = st.slider("Sampling Temperature", 0.5, 1.5, 0.85, 0.05)
        generate_btn = st.button("Generate Protein", type="primary", use_container_width=True)

    with col2:
        if generate_btn:
            with st.spinner("Sampling latent space & decoding sequence…"):
                try:
                    model, tokenizer, _, device = _cached_model()
                    time.sleep(0.4)
                    seq = generate_protein_sequence(
                        model, tokenizer, seq_len, protein_type,
                        temperature=temperature, device=device,
                    )
                    pid = f"Gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    store_generated_protein(seq, protein_type, pid)
                    save_generated_to_files(seq, pid, protein_type)
                    st.success(
                        f"Generated **{pid}** ({len(seq)} residues) — "
                        "available in **Protein Analysis**"
                    )
                    st.rerun()
                except Exception as ex:
                    st.error(f"Generation failed: {ex}")

        if st.session_state.get("generated_protein"):
            gp = st.session_state.generated_protein
            seq = gp["sequence"]
        elif st.session_state.last_sequence:
            seq = st.session_state.last_sequence
        else:
            seq = ""

        if seq:
            st.markdown("**Generated Sequence**")
            st.markdown(f'<div class="seq-box">{seq}</div>', unsafe_allow_html=True)
            stats = analyze_sequence(seq)
            render_metric_grid([
                ("Length", stats["length"]),
                ("Hydrophobicity", f"{stats['hydrophobicity']:.3f}"),
                ("Entropy", f"{stats['entropy']:.3f}"),
                ("Dominant AA", stats["dominant_aa"]),
            ])
            st.markdown(
                " ".join(
                    f'<span class="metric-pill">{aa}: {v*100:.1f}%</span>'
                    for aa, v in sorted(stats["composition"].items(), key=lambda x: -x[1])[:6]
                ),
                unsafe_allow_html=True,
            )


def page_metrics():
    st.markdown('<p class="section-header">Research Metrics Dashboard</p>', unsafe_allow_html=True)
    metrics = load_research_metrics()

    row1 = st.columns(3)
    row2 = st.columns(3)
    gauge_specs = [
        (metrics.get("Novelty_Score", 1.0), "Novelty Score", 1.0, "#69f0ae"),
        (metrics.get("Diversity_Score", 0.93), "Diversity Score", 1.0, "#00e5ff"),
        (metrics.get("ProtBERT_Alignment", 0.9888), "ProtBERT Alignment", 1.0, "#7c4dff"),
        (metrics.get("Sequence_Entropy", 4.18) / 5.0, "Sequence Entropy (norm.)", 1.0, "#ffab40"),
        (metrics.get("Foldability_Score", 4.4) / 10.0, "Foldability Score", 1.0, "#18ffff"),
        (metrics.get("Biological_Realism", 0.9), "Biological Realism", 1.0, "#ea80fc"),
    ]
    for col, (val, title, mx, color) in zip(row1 + row2, gauge_specs):
        with col:
            st.plotly_chart(plot_gauge(val, title, mx, color), use_container_width=True)

    st.markdown("#### Metric Progress Bars")
    bars = {
        "Novelty": metrics.get("Novelty_Score", 1.0),
        "Diversity": metrics.get("Diversity_Score", 0.93),
        "ProtBERT": metrics.get("ProtBERT_Alignment", 0.99),
        "Realism": metrics.get("Biological_Realism", 0.9),
        "PCA Variance": metrics.get("Latent_PCA_Variance", 0.8),
    }
    for name, val in bars.items():
        pct = min(float(val), 1.0) * 100
        st.markdown(
            f"""
<div class="progress-row">
  <div class="progress-label">{name}: {val:.4f}</div>
  <div class="progress-track">
    <div class="progress-fill" style="width:{pct:.1f}%"></div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )


def page_latent():
    st.markdown('<p class="section-header">Latent Space Visualization</p>', unsafe_allow_html=True)
    n_pts = st.slider("Number of latent points", 100, 800, 400, 50)

    with st.spinner("Computing PCA & t-SNE…"):
        latent_df = generate_latent_space_data(n_pts)
        pca_df = compute_pca_2d(latent_df)
        tsne_df = compute_tsne_2d(latent_df)

    c1, c2 = st.columns(2)
    with c1:
        var = pca_df["variance_explained"].iloc[0] if "variance_explained" in pca_df else 0.8
        st.plotly_chart(
            plot_latent_scatter(pca_df, "PC1", "PC2", f"PCA Projection (var. explained ≈ {var:.2%})"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            plot_latent_scatter(tsne_df, "tSNE1", "tSNE2", "t-SNE Clustering by Protein Type"),
            use_container_width=True,
        )


def page_analysis():
    st.markdown('<p class="section-header">Protein Sequence Analysis</p>', unsafe_allow_html=True)

    keys, labels, lookup = build_analysis_sequence_options()
    has_latest = st.session_state.get("generated_protein") is not None

    btn_col, sel_col = st.columns([1, 2])
    with btn_col:
        if st.button(
            "Analyze Latest Generated Protein",
            type="primary",
            use_container_width=True,
            disabled=not has_latest,
        ):
            st.session_state.force_analysis_latest = True
            st.session_state.analysis_select_key = LATEST_PROTEIN_KEY
            st.rerun()

    with sel_col:
        if not has_latest:
            st.info("Generate a protein on the **Protein Generation** page to analyze it here.")

        if st.session_state.get("force_analysis_latest") and LATEST_PROTEIN_KEY in keys:
            st.session_state.analysis_select_key = LATEST_PROTEIN_KEY
            st.session_state.force_analysis_latest = False

        current_key = st.session_state.get("analysis_select_key", keys[0])
        if current_key not in keys:
            current_key = keys[0]
        default_index = keys.index(current_key)

        choice_label = st.selectbox("Select sequence", labels, index=default_index)
        selected_key = keys[labels.index(choice_label)]
        st.session_state.analysis_select_key = selected_key

    if selected_key == CUSTOM_INPUT_KEY:
        default_seq = ""
        if has_latest:
            default_seq = st.session_state.generated_protein["sequence"]
        sequence = st.text_area(
            "Enter protein sequence",
            default_seq or (
                "MIDSLIRGILEAEGRVVDDVLRADFVLDRLTLSEERIVAKGAVDAGVEIVARAGRPEKAAVILGVASGMPSLALRPEMRSFMSSLSLADEFLHVKAAWDWRHPAAAAAGFALGMGTAALTYLTGIARQAFISRPDDMGVWLKRHGMFEVVYRAVDAGVALPNLTLEWQ"
            ),
            height=120,
        )
        meta = None
    else:
        sequence = lookup.get(selected_key, "")
        meta = (
            st.session_state.generated_protein
            if selected_key == LATEST_PROTEIN_KEY
            else next(
                (h for h in st.session_state.generated_history if f"hist_{h['id']}" == selected_key),
                None,
            )
        )

    if sequence:
        show_highlight = selected_key == LATEST_PROTEIN_KEY and meta is not None
        render_full_sequence_analysis(
            sequence,
            meta=meta if show_highlight else None,
        )
    elif selected_key != CUSTOM_INPUT_KEY:
        st.warning("No sequence found for the selected entry.")


def page_download():
    st.markdown('<p class="section-header">Download & Export</p>', unsafe_allow_html=True)

    df = load_generated_proteins()
    fasta_raw = load_fasta_content()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Stored Proteins", len(df))
    with c2:
        st.metric("FASTA Records", fasta_raw.count(">") if fasta_raw else 0)
    with c3:
        st.metric("Avg Length", int(df["Length"].mean()) if len(df) and "Length" in df else 0)

    if len(df) > 0:
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            data=export_csv_download(df),
            file_name="generated_proteins.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if fasta_raw:
        st.download_button(
            "Download FASTA",
            data=fasta_raw.encode("utf-8"),
            file_name="generated_proteins.fasta",
            mime="text/plain",
            use_container_width=True,
        )

    gp = st.session_state.get("generated_protein")
    if gp and gp.get("sequence"):
        st.markdown("#### Export Last Generated Protein")
        seq = gp["sequence"]
        pid = gp.get("id", "latest")
        ptype = gp.get("type", "Unknown")
    elif st.session_state.get("last_sequence"):
        st.markdown("#### Export Last Generated Protein")
        seq = st.session_state.last_sequence
        pid = st.session_state.get("last_id", "latest")
        ptype = st.session_state.get("last_type", "Unknown")
    else:
        seq = ""

    if seq:
        st.download_button(
            "Export Last as FASTA",
            data=export_fasta_download([(pid, ptype, seq)]),
            file_name=f"{pid}.fasta",
            mime="text/plain",
        )


init_session_storage()

# ── Sidebar navigation ─────────────────────────────────────────────────────────
PAGES = {
    "Home": page_home,
    "Model Architecture": page_architecture,
    "Protein Generation": page_generation,
    "Metrics Dashboard": page_metrics,
    "Latent Space": page_latent,
    "Protein Analysis": page_analysis,
    "Download": page_download,
}

with st.sidebar:
    st.markdown(
        '<h1 style="margin:0;padding:0;">🧬 ProteinVAE</h1>',
        unsafe_allow_html=True,
    )
    st.caption("Hybrid Transformer-BiLSTM VAE · Research Deployment")
    st.divider()
    page = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    try:
        _, _, cfg, dev = _cached_model()
        st.success(f"Model ready ({dev})")
    except Exception as e:
        st.error("Model not loaded")
        st.caption(str(e)[:80])
    st.markdown(
        """
<div class="sidebar-info">
<p><strong>Research Deployment</strong></p>
<ul>
<li><code>protein_vae_final.pth</code></li>
<li>128D latent sampling</li>
<li>Conditional generation</li>
</ul>
</div>
        """,
        unsafe_allow_html=True,
    )

# ── Render selected page ─────────────────────────────────────────────────────────
PAGES[page]()
