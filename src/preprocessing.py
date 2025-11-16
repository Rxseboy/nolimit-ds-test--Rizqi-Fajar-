"""
Tokenization & preprocessing utilities.
Uses the CardiffNLP tokenizer with `max_length=256` exactly as used in notebook.
"""

from transformers import AutoTokenizer
from typing import Dict


from .config import model_cfg


# Load tokenizer once globally (best practice)
tokenizer = AutoTokenizer.from_pretrained(model_cfg.MODEL_NAME)


def tokenize_batch(batch: Dict):
	"""Tokenize text batch using model tokenizer."""
	return tokenizer(
		batch["text"],
		truncation=True,
		padding="max_length",
		max_length=model_cfg.MAX_LENGTH,
	)


def preprocess_dataset(ds):
	"""
	Apply tokenization over dataset splits.
	Returns a new DatasetDict.
	"""
	return ds.map(tokenize_batch, batched=True)