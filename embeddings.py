"""Dense embedding via SamilPwC-AXNode-GenAI/PwC-Embedding_expr (multilingual-e5-large-instruct finetune)."""
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "SamilPwC-AXNode-GenAI/PwC-Embedding_expr"

_device = "mps" if torch.backends.mps.is_available() else "cpu"
_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME).to(_device).eval()
    return _tokenizer, _model


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def embed(texts: list[str]) -> np.ndarray:
    """Returns L2-normalized embeddings, shape (len(texts), hidden_size)."""
    tokenizer, model = _load()
    batch = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(_device)
    out = model(**batch)
    pooled = _mean_pool(out.last_hidden_state, batch["attention_mask"])
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.cpu().numpy()
