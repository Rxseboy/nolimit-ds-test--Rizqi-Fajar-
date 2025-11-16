"""
Model training module using Hugging Face Trainer.
This follows the exact training process from the notebook but structured professionally.
"""


import os
import evaluate
import numpy as np
from transformers import (
AutoModelForSequenceClassification,
TrainingArguments,
Trainer,
)


from .config import model_cfg, train_cfg, paths
from .preprocessing import tokenizer




def compute_metrics(eval_pred):
	"""
	Compute accuracy, precision, recall, F1 (weighted).
	Matches notebook metrics exactly.
	"""
	logits, labels = eval_pred
	preds = np.argmax(logits, axis=-1)

	accuracy = evaluate.load("accuracy").compute(predictions=preds, references=labels)["accuracy"]
	precision = evaluate.load("precision").compute(predictions=preds, references=labels, average="weighted")["precision"]
	recall = evaluate.load("recall").compute(predictions=preds, references=labels, average="weighted")["recall"]
	f1 = evaluate.load("f1").compute(predictions=preds, references=labels, average="weighted")["f1"]

	return {
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"f1": f1,
	}




def load_model():
	"""Load CardiffNLP sentiment model with correct label count."""
	model = AutoModelForSequenceClassification.from_pretrained(
		model_cfg.MODEL_NAME,
		num_labels=model_cfg.NUM_LABELS,
	)
	return model




def build_training_args():
	"""Factory for HF TrainingArguments clean configuration."""
	return TrainingArguments(
		output_dir=train_cfg.OUTPUT_DIR,
		evaluation_strategy="epoch",
		save_strategy="epoch",
		learning_rate=train_cfg.LR,
		per_device_train_batch_size=train_cfg.TRAIN_BATCH,
		per_device_eval_batch_size=train_cfg.EVAL_BATCH,
		num_train_epochs=train_cfg.EPOCHS,
		weight_decay=train_cfg.WEIGHT_DECAY,
	)

def train_model(dataset_dict):
	"""
	Train the model using Trainer API.
	Saves trained model + tokenizer to trained_model directory.
	"""
	model = load_model()
	args = build_training_args()

	trainer = Trainer(
		model=model,
		args=args,
		train_dataset=dataset_dict["train"],
		eval_dataset=dataset_dict["validation"],
		compute_metrics=compute_metrics,
		tokenizer=tokenizer,
	)

	trainer.train()
	trainer.save_model(paths.TRAINED_MODEL_DIR)
	tokenizer.save_pretrained(paths.TRAINED_MODEL_DIR)

	return trainer