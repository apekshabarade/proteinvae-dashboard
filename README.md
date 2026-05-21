# ProteinVAE – AI-Driven De Novo Protein Generation Dashboard

Professional Streamlit dashboard for the **Hybrid Transformer-BiLSTM ProteinVAE** research deployment.

## Prerequisites

- Python 3.10+
- Deployment assets from `ProteinVAE_Deployment.zip` (extracted)

## Setup

```bash
cd protein_dashboard
pip install -r requirements.txt
```

Ensure these files exist (copied from the zip or `../deployment/`):

| File | Location |
|------|----------|
| `protein_vae_final.pth` | `models/` |
| `tokenizer.pkl` | `data/` |
| `model_config.json` | `data/` |
| `generated_proteins.csv` | `data/` |
| `generated_proteins.fasta` | `data/` |
| `research_metrics.json` | `data/` |
| `conditional_labels.json` | `data/` |

The app also reads from `../deployment/` automatically if `data/` copies are missing.

## Run locally

```bash
cd protein_dashboard
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## Dashboard sections

1. **Home** – Project overview & key statistics  
2. **Model Architecture** – Encoder → Latent → Decoder pipeline  
3. **Protein Generation** – Live inference with type & length controls  
4. **Metrics Dashboard** – Novelty, ProtBERT alignment, gauges  
5. **Latent Space** – PCA & t-SNE visualizations  
6. **Protein Analysis** – AA composition & hydrophobicity  
7. **Download** – Export CSV / FASTA  

## Project structure

```
protein_dashboard/
├── app.py              # Main Streamlit application
├── model_loader.py     # Model load, inference, caching
├── utils.py            # Metrics, Plotly charts, analysis
├── requirements.txt
├── models/
│   ├── protein_vae.py  # Architecture (matches .pth weights)
│   └── protein_vae_final.pth
├── data/               # Config, tokenizer, exports
└── assets/
```

## GPU support

CUDA is used automatically when available. CPU fallback is supported.
