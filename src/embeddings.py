"""
Embedding utilities using CardiffNLP RoBERTa
"""


import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification


from .config import model_cfg, paths
from .preprocessing import tokenizer




__all__ = [
	"load_trained_model",
	"get_embeddings",
	"save_embeddings",
	"load_embeddings",
]




_cached_model = None


def load_trained_model():
	"""Lazy-load fine‑tuned model for embedding extraction."""
	global _cached_model
	if _cached_model is None:
		_cached_model = AutoModelForSequenceClassification.from_pretrained(
			paths.TRAINED_MODEL_DIR,
			output_hidden_states=True,
		).to(model_cfg.DEVICE)
		_cached_model.eval()
	return _cached_model




def get_embeddings(texts, batch_size: int = 16):
	"""Extract CLS embeddings for a list of texts.
	Returns a NumPy matrix of shape (N, H).
	"""
	model = load_trained_model()
	model.eval()
	all_vecs = []

	# Use a simple collate_fn to keep batches as lists of strings (works for tokenizer)
	dataloader = DataLoader(texts, batch_size=batch_size, collate_fn=lambda x: x)

	with torch.no_grad():
		for batch in dataloader:
			enc = tokenizer(
				batch,
				return_tensors="pt",
				padding=True,
				truncation=True,
				max_length=model_cfg.MAX_LENGTH,
			).to(model_cfg.DEVICE)

			out = model(**enc)
			cls_vec = out.hidden_states[-1][:, 0, :].cpu().numpy()
			all_vecs.append(cls_vec)

	return np.vstack(all_vecs)


def save_embeddings(embeddings: np.ndarray, path: str) -> None:
	"""Save embeddings (NumPy array) to the given path using .npy format."""
	dirname = os.path.dirname(path)
	if dirname:
		os.makedirs(dirname, exist_ok=True)
	# Use allow_pickle=False for safety; callers should pass a pure numeric array.
	np.save(path, embeddings)


def load_embeddings(path: str) -> np.ndarray:
	"""Load embeddings saved with save_embeddings and return as a NumPy array."""
	# Let exceptions propagate (FileNotFoundError, etc.) so callers can handle them.
	return np.load(path, allow_pickle=False)