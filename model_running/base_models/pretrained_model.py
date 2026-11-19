import torch
import torch.nn as nn
from transformers import AutoModel


class PretrainedModel(nn.Module):
    def __init__(
        self,
        num_punc,
        num_cap,
        model_name: str,
        cache_dir: str,
        freeze_model: bool,
        dropout_rate: float
    ):
        super(PretrainedModel, self).__init__()
        
        self.encoder = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            output_attentions=True
        )

        if freeze_model:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.punc = torch.nn.Linear(self.encoder.config.hidden_size, num_punc)
        self.cap = torch.nn.Linear(self.encoder.config.hidden_size, num_cap)

    def forward(self, **input):
        """Forward pass through the model."""
        encoder_output = self.encoder(**input)

        cls_hidden_state = encoder_output.last_hidden_state

        attentions = encoder_output.attentions
        attn_cls = attentions[-1].mean(1)[0, 0, :]

        dropped_output = self.dropout(cls_hidden_state)
        punc_logits = self.punc(dropped_output)
        cap_logits = self.cap(dropped_output)

        return punc_logits, cap_logits, attn_cls
