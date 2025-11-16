"""
Dataset loading and balancing utilities for Tweet Sentiment Project.
Implements the same logic used in the notebook: undersampling each label.
"""


import numpy as np
from datasets import load_dataset, DatasetDict
from typing import Tuple


from .config import model_cfg




def load_raw_dataset() -> DatasetDict:
	"""Load the TweetEval sentiment dataset."""
	return load_dataset("cardiffnlp/tweet_eval", "sentiment")




def balance_split(split, target_size: int) -> DatasetDict:
	"""
	Undersample dataset split so each class has `target_size` items.
	Matches the notebook's balancing strategy.
	"""
	labels = split["label"]
	unique_labels = np.unique(labels)

	balanced_indices = []
	for label in unique_labels:
		indices = [i for i, l in enumerate(labels) if l == label]
		chosen = np.random.choice(indices, target_size, replace=False)
		balanced_indices.extend(chosen)

	return split.select(balanced_indices)




def load_balanced_dataset(train_size=1500, val_size=100, test_size=500) -> DatasetDict:
	"""
	Create a balanced DatasetDict identical to the notebook.
	"""
	raw = load_raw_dataset()

	train = balance_split(raw["train"], train_size)
	val = balance_split(raw["validation"], val_size)
	test = balance_split(raw["test"], test_size)

	return DatasetDict({
		"train": train,
		"validation": val,
		"test": test
	})