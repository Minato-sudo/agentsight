import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType

class VanillaBaselineModel(nn.Module):
    """
    A simple baseline deep learning model that takes the entire trajectory
    and predicts if a hallucination occurred.
    It does NOT use the AgentSight Context Encoder or Dual Heads.
    """
    def __init__(self, encoder_name="microsoft/deberta-v3-base"):
        super().__init__()
        
        base_encoder = AutoModel.from_pretrained(encoder_name, torch_dtype=torch.float32)
        base_encoder.gradient_checkpointing_enable()
        
        # Apply LoRA to the base encoder
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query_proj", "value_proj"]
        )
        self.encoder = get_peft_model(base_encoder, peft_config)
        self.encoder.print_trainable_parameters()
        
        enc_dim = self.encoder.config.hidden_size
        
        # Simple binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(enc_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1) # Raw logits for BCEWithLogitsLoss
        )

    def forward(self, input_ids, attention_mask):
        # Forward pass through DeBERTa (N_steps, max_len)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the representation of the [CLS] token (the first token) for each step
        cls_repr = outputs.last_hidden_state[:, 0, :] # Shape: (N_steps, enc_dim)
        
        # MEAN POOLING across all steps to create a single Trajectory Representation
        # This guarantees gradients flow backwards through ALL steps, unlike max() pooling!
        traj_repr = cls_repr.mean(dim=0).unsqueeze(0) # Shape: (1, enc_dim)
        
        # Predict hallucination for the entire sequence
        logits = self.classifier(traj_repr).squeeze(-1) # Shape: (1,)
        return logits
