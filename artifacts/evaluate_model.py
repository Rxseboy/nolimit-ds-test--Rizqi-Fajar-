"""
Evaluation utilities for sentiment model.
"""


from transformers import Trainer
from .model_training import compute_metrics




def evaluate_model(trainer: Trainer, eval_dataset):
	"""Run evaluation using the Trainer instance."""
	results = trainer.evaluate(eval_dataset=eval_dataset)
	return results