"""
ProteinVAE – Hybrid Transformer-BiLSTM Variational Autoencoder
Architecture aligned with protein_vae_final.pth deployment weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinEmbedding(nn.Module):
    """Token + positional embedding with layer norm."""

    def __init__(self, vocab_size: int, embed_dim: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_encoding = PositionalEncoding(embed_dim, max_len)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
        x = self.position_encoding(x)
        return self.dropout(self.layer_norm(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBiLSTMEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.embedding = ProteinEmbedding(vocab_size, embed_dim, max_len, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.bilstm = nn.LSTM(
            embed_dim, hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = x.eq(0)
        emb = self.embedding(x)
        t_out = self.transformer_encoder(emb, src_key_padding_mask=mask)
        lstm_out, _ = self.bilstm(t_out)
        pooled = lstm_out.mean(dim=1)
        fused = self.fusion(pooled)
        return self.mu_layer(fused), self.logvar_layer(fused)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.embedding = ProteinEmbedding(vocab_size, embed_dim, max_len, dropout)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, vocab_size),
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt_emb = self.embedding(tgt)
        out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.output_layer(out)


class ProteinVAE(nn.Module):
    """Hybrid Transformer-BiLSTM VAE for protein sequence generation."""

    def __init__(self, config: dict):
        super().__init__()
        vocab_size = config["vocab_size"]
        embed_dim = config["embed_dim"]
        hidden_dim = config["hidden_dim"]
        latent_dim = config["latent_dim"]
        num_layers = config.get("num_layers", 3)
        num_heads = config.get("num_heads", 8)
        max_len = config.get("max_seq_len", 2048)  # position encoding buffer size
        dropout = config.get("dropout", 0.1)

        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

        self.encoder = TransformerBiLSTMEncoder(
            vocab_size, embed_dim, hidden_dim, latent_dim, num_layers, num_heads, dropout, max_len
        )
        self.decoder = TransformerDecoder(
            vocab_size, embed_dim, latent_dim, hidden_dim, num_layers, num_heads, dropout, max_len
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode_step(self, z: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory = self.decoder.latent_projection(z).unsqueeze(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        return self.decoder(tgt, memory, tgt_mask)

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        max_len: int,
        sos_idx: int = 1,
        eos_idx: int = 2,
        temperature: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """Autoregressive generation with anti-repetition decoding."""
        self.eval()
        device = z.device
        batch = z.size(0)
        generated = torch.full((batch, 1), sos_idx, dtype=torch.long, device=device)
        memory = self.decoder.latent_projection(z).unsqueeze(1)

        for _ in range(max_len - 1):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                generated.size(1), device=device
            )
            tgt_emb = self.decoder.embedding(generated)
            out = self.decoder.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.decoder.output_layer(out[:, -1, :]) / temperature

            if repetition_penalty != 1.0 and generated.size(1) > 1:
                for b in range(batch):
                    for token in set(generated[b].tolist()):
                        if token not in (0, sos_idx):
                            logits[b, token] /= repetition_penalty

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_idx).all():
                break

        return generated
