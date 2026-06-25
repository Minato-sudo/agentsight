import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType


class AgentSightModel(nn.Module):
    """
    AgentSight: Step-level hallucination classifier for agent trajectories.

    Architecture
    ────────────
    1.  DeBERTa-v3-base encoder (frozen backbone) with LoRA adapters
        injected into all attention projections.
    2.  Transformer Context-Encoder: a shallow 3-layer Transformer that reads
        the sequence of per-step CLS representations and lets steps attend
        to each other — capturing error-accumulation dynamics.
    3.  Projection head → binary hallucination logit per step.

    Input
    ─────
    A single trajectory encoded as N (input_ids, attention_mask) pairs, one
    per agent step.  Each step's text is a concatenated string:
        [QUERY] <task> [THOUGHT] <reasoning> [ACTION] <calls> [OBS] <results>

    The ablation study (A1) shows that removing [ACTION]+[OBS] drops step
    localisation accuracy by ~28.8 points, confirming that tool execution
    content is the dominant signal.

    Output
    ──────
    hal_logits: (N,) float tensor — raw logits, NOT sigmoids.
    Apply torch.sigmoid() at inference; use BCEWithLogitsLoss during training.
    """

    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-base",
        hidden_dim: int = 256,
        dropout: float = 0.2,
        n_context_layers: int = 3,
        lora_r: int = 16,
        lora_alpha: int = 64,
    ):
        super().__init__()

        # ── 1. Backbone ───────────────────────────────────────────────────
        base_encoder = AutoModel.from_pretrained(encoder_name, dtype=torch.float32)
        base_encoder.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],
        )
        self.encoder = get_peft_model(base_encoder, peft_config)
        self.encoder.print_trainable_parameters()

        enc_dim = self.encoder.config.hidden_size  # 768 for deberta-v3-base

        # ── 2. Cross-step Context Encoder ─────────────────────────────────
        ctx_layer = nn.TransformerEncoderLayer(
            d_model=enc_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for better gradient flow
        )
        self.context_encoder = nn.TransformerEncoder(
            ctx_layer, num_layers=n_context_layers,
            enable_nested_tensor=False,
        )

        # ── 3. Projection + Classification head ───────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            # NOTE: no Sigmoid — use BCEWithLogitsLoss for numerical stability
        )

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args
        ────
        input_ids      : (N, seq_len) — one row per trajectory step
        attention_mask : (N, seq_len)

        Returns
        ───────
        hal_logits : (N,) — raw pre-sigmoid hallucination logits
        """
        # Encode each step independently with DeBERTa
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        step_repr = out.last_hidden_state[:, 0, :].to(torch.float32)  # (N, enc_dim)

        # Cross-step attention: treat the N steps as a sequence of length N
        # (add and remove a dummy batch dim of 1)
        context = self.context_encoder(step_repr.unsqueeze(0)).squeeze(0)  # (N, enc_dim)

        fused = self.fusion(context)                     # (N, hidden_dim)
        hal_logits = self.cls_head(fused).squeeze(-1)   # (N,)

        return hal_logits
