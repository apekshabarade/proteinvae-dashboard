"""
Utility functions: sequence analysis, metrics, Plotly visualizations.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Kyte-Doolittle hydrophobicity
HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEPLOYMENT_DIR = ROOT.parent / "deployment"


def get_data_path(filename: str) -> Path:
    p = DATA_DIR / filename
    if p.exists():
        return p
    return DEPLOYMENT_DIR / filename


def sequence_length(sequence: str) -> int:
    return len(sequence)


def amino_acid_composition(sequence: str) -> dict[str, float]:
    seq = sequence.upper()
    n = len(seq) or 1
    counts = {aa: seq.count(aa) / n for aa in STANDARD_AA}
    return counts


def hydrophobicity_estimate(sequence: str) -> float:
    vals = [HYDROPHOBICITY.get(aa, 0.0) for aa in sequence.upper() if aa in HYDROPHOBICITY]
    return float(np.mean(vals)) if vals else 0.0


def sequence_entropy(sequence: str) -> float:
    seq = sequence.upper()
    n = len(seq) or 1
    freqs = np.array([seq.count(aa) for aa in STANDARD_AA], dtype=float)
    freqs = freqs[freqs > 0] / n
    return float(-np.sum(freqs * np.log2(freqs + 1e-12)))


def analyze_sequence(sequence: str) -> dict:
    comp = amino_acid_composition(sequence)
    return {
        "length": len(sequence),
        "composition": comp,
        "hydrophobicity": hydrophobicity_estimate(sequence),
        "entropy": sequence_entropy(sequence),
        "dominant_aa": max(comp, key=comp.get) if comp else "—",
        "confidence": min(0.99, 0.75 + sequence_entropy(sequence) / 10),
    }


def load_generated_proteins() -> pd.DataFrame:
    path = get_data_path("generated_proteins.csv")
    if path.exists():
        df = pd.read_csv(path)
        if "Sequence" in df.columns:
            return df
        if "sequence" in df.columns:
            df = df.rename(columns={"sequence": "Sequence"})
        return df
    return pd.DataFrame(columns=["Protein_ID", "Sequence", "Length"])


def load_fasta_content() -> str:
    path = get_data_path("generated_proteins.fasta")
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def generate_latent_space_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic latent embeddings for PCA/t-SNE (demo visualization)."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(["Enzyme", "Membrane", "Binding"], size=n, p=[0.4, 0.35, 0.25])
    z = rng.standard_normal((n, 128))
    for i, lab in enumerate(labels):
        if lab == "Enzyme":
            z[i, :32] += 0.8
        elif lab == "Membrane":
            z[i, 32:64] += 0.8
        else:
            z[i, 64:96] += 0.8
    return pd.DataFrame(z, columns=[f"z{i}" for i in range(128)]).assign(protein_type=labels)


def compute_pca_2d(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    from sklearn.decomposition import PCA

    cols = [c for c in df.columns if c.startswith("z")]
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(df[cols].values)
    out = df.copy()
    out["PC1"] = coords[:, 0]
    out["PC2"] = coords[:, 1]
    out["variance_explained"] = float(pca.explained_variance_ratio_.sum())
    return out


def compute_tsne_2d(df: pd.DataFrame, perplexity: int = 30) -> pd.DataFrame:
    from sklearn.manifold import TSNE

    cols = [c for c in df.columns if c.startswith("z")]
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(df) - 1), random_state=42)
    coords = tsne.fit_transform(df[cols].values)
    out = df.copy()
    out["tSNE1"] = coords[:, 0]
    out["tSNE2"] = coords[:, 1]
    return out


def plot_aa_frequency(sequence: str) -> go.Figure:
    comp = amino_acid_composition(sequence)
    df = pd.DataFrame({"Amino Acid": list(comp.keys()), "Frequency": list(comp.values())})
    fig = px.bar(
        df, x="Amino Acid", y="Frequency",
        color="Frequency", color_continuous_scale="Viridis",
        title="Amino Acid Frequency Distribution",
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0f7fa"), height=380,
    )
    return fig


def plot_latent_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    fig = px.scatter(
        df, x=x_col, y=y_col, color="protein_type",
        color_discrete_map={"Enzyme": "#00e5ff", "Membrane": "#7c4dff", "Binding": "#69f0ae"},
        hover_data=["protein_type"], title=title, opacity=0.75,
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,40,0.8)",
        font=dict(color="#e0f7fa"), height=420,
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
    )
    return fig


def plot_gauge(value: float, title: str, max_val: float = 1.0, color: str = "#00e5ff") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 14, "color": "#b0bec5"}},
            number={"font": {"size": 28, "color": "#e0f7fa"}},
            gauge={
                "axis": {"range": [0, max_val], "tickcolor": "#546e7a"},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0.3)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, max_val * 0.5], "color": "rgba(0,229,255,0.15)"},
                    {"range": [max_val * 0.5, max_val * 0.8], "color": "rgba(124,77,255,0.2)"},
                    {"range": [max_val * 0.8, max_val], "color": "rgba(105,240,174,0.25)"},
                ],
            },
        )
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=220,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def plot_architecture_flow() -> go.Figure:
    """Encoder → Latent → Decoder pipeline diagram."""
    fig = go.Figure()
    nodes = [
        (0, 2, "Input Sequence"),
        (2, 2, "Transformer Encoder"),
        (4, 2, "BiLSTM"),
        (6, 2, "μ / log σ²"),
        (8, 2, "Latent z (128D)"),
        (10, 2, "Transformer Decoder"),
        (12, 2, "Protein Output"),
    ]
    for x, y, label in nodes:
        fig.add_shape(
            type="rect", x0=x - 0.7, y0=y - 0.35, x1=x + 0.7, y1=y + 0.35,
            fillcolor="rgba(0,229,255,0.2)", line=dict(color="#00e5ff", width=2),
        )
        fig.add_annotation(x=x, y=y, text=label, showarrow=False, font=dict(size=10, color="#e0f7fa"))
    for i in range(len(nodes) - 1):
        fig.add_annotation(
            x=(nodes[i][0] + nodes[i + 1][0]) / 2, y=2.6,
            ax=nodes[i][0] + 0.7, ay=2, axref="x", ayref="y",
            xref="x", yref="y", showarrow=True, arrowhead=2, arrowcolor="#7c4dff",
        )
    fig.update_layout(
        title="ProteinVAE Generation Pipeline",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-1, 13]),
        yaxis=dict(visible=False, range=[0.5, 3.5]),
        height=280, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def sequence_heatmap_data(sequence: str, window: int = 10) -> pd.DataFrame:
    """Sliding-window hydrophobicity for sequence visualization."""
    seq = sequence.upper()
    scores = []
    for i in range(len(seq)):
        w = seq[max(0, i - window // 2) : i + window // 2 + 1]
        scores.append(hydrophobicity_estimate(w))
    return pd.DataFrame({"Position": range(1, len(seq) + 1), "Hydrophobicity": scores})


def plot_sequence_profile(sequence: str) -> go.Figure:
    df = sequence_heatmap_data(sequence)
    fig = px.area(df, x="Position", y="Hydrophobicity", title="Sequence Hydrophobicity Profile")
    fig.update_traces(fillcolor="rgba(0,229,255,0.3)", line_color="#00e5ff")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,40,0.8)",
        font=dict(color="#e0f7fa"), height=300,
    )
    return fig


def export_fasta_download(sequences: list[tuple[str, str, str]]) -> bytes:
    buf = io.StringIO()
    for pid, ptype, seq in sequences:
        buf.write(f">{pid}|{ptype}|len={len(seq)}\n")
        for i in range(0, len(seq), 80):
            buf.write(seq[i : i + 80] + "\n")
    return buf.getvalue().encode("utf-8")


def export_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
