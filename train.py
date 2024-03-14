#!/bin/python3

import sys
import os

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from torchmetrics.functional.classification import binary_confusion_matrix

from tokenizers.normalizers import Replace, Sequence as nSequence
from tokenizers.pre_tokenizers import ByteLevel, Punctuation, Sequence as pSequence, Whitespace

import transformers
from transformers import AutoTokenizer

import evaluate
import datasets


VOCAB_SIZE = 2 ** 17
FORMULA_MAX_LENGTH_IN_TOKENS = 1024
TOKENIZER_BATCH_SIZE = 2 ** 10

GRADIENT_ACCUMULATION_STEPS = 8
DEVICE_TRAIN_BATCH_SIZE = 8
DEVICE_VAL_BATCH_SIZE = 64

TRAIN_EPOCHS = 10
FINETUNE_EPOCHS = 5

torch.set_num_threads(16)


class FormulaDataset(Dataset):
	def __init__(self, formulas, labels, stage):
		self.formulas = formulas
		self.labels = labels
		self.stage = stage

	def setup(self, tokenizer, max_length):
		self.samples = []
		formulas_cnt = len(self.formulas)
		indices_to_delete = []
		cur_idx = 0
		
		for i in tqdm(range(0, formulas_cnt, TOKENIZER_BATCH_SIZE), f"tokenizing {self.stage} dataset"):
			formulas_batch = self.formulas[i : min(formulas_cnt, i + TOKENIZER_BATCH_SIZE)]
			labels_batch = self.labels[i : min(formulas_cnt, i + TOKENIZER_BATCH_SIZE)]
			
			formulas_batch = tokenizer(formulas_batch)
			formulas_batch = [dict(zip(formulas_batch, i)) for i in zip(*formulas_batch.values())]

			for formula, label in zip(formulas_batch, labels_batch):
				cur_idx += 1
				
				if len(formula["input_ids"]) > max_length:
					indices_to_delete.append(cur_idx - 1)
					continue

				self.samples.append({ **formula, "label": label })

		for i in reversed(indices_to_delete):
			self.formulas.pop(i)
			self.labels.pop(i)

		print(f"{self.stage} dataset set up")
		
		sat_cnt = sum(s["label"] for s in self.samples)
		unsat_cnt = len(self.samples) - sat_cnt
		
		print(f"{len(self.samples)} overall | {sat_cnt} sat | {unsat_cnt} unsat | sat frac: {sat_cnt / len(self.samples)}")
		print()

	def get_iterator_for_tokenizer(self):
		formulas_cnt = len(self.formulas)
		
		for i in range(0, formulas_cnt, TOKENIZER_BATCH_SIZE):
			yield self.formulas[i : min(formulas_cnt, i + TOKENIZER_BATCH_SIZE)]

	def get_iterator(self):
		formulas_cnt = len(self.formulas)

		for formula, label in zip(self.formulas, self.labels):
			yield { "text": formula, "label": label }

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		return self.samples[index]


def get_file_paths(stage, dataset_labels):
	file_paths = []
	
	for ds in dataset_labels:
		with open(f"dataset/{ds}/__meta/{stage}", "r") as f:
			file_paths += [f"dataset/{ds}/{file_path.strip()}" for file_path in f.readlines()]

	return file_paths


def get_dataset(stage, dataset_labels):
	file_paths = get_file_paths(stage, dataset_labels)
	if stage == "train" and "SHRINK_TRAIN" in os.environ:
		file_paths = file_paths[:int(os.environ["SHRINK_TRAIN"])]
	
	formulas, labels = [], []
	for file_path in tqdm(file_paths, "reading files with formulas"):
		with open(file_path, "r") as f:
			formula = f.read()
		
		label = 0 if file_path.endswith("-unsat") else 1

		formulas.append(formula)
		labels.append(label)

	return FormulaDataset(formulas, labels, stage)


accuracy_metric = evaluate.load("accuracy")
roc_auc_metric = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
	def calc_softmax(scores):
		scores = np.exp(scores)
		return scores / scores.sum(axis=-1, keepdims=True)

	scores, labels = eval_pred
	predictions = np.argmax(scores, axis=-1)
	scores = calc_softmax(scores)[:, 1]

	accuracy = accuracy_metric.compute(references=labels, predictions=predictions)

	try:
		roc_auc = roc_auc_metric.compute(references=labels, prediction_scores=scores)

	except ValueError as e:
		roc_auc = float('nan')

	return {
		"acc": accuracy["accuracy"],
		"roc_auc": roc_auc if type(roc_auc) is float else roc_auc["roc_auc"],
	}


def print_confusion_matrix(conf_mat):
	conf_mat = np.around(conf_mat, 7)
	
	print()
	print("        +-------+-----------+-----------+")
	print("       ", "|", "unsat", "|", str(conf_mat[0][0]).rjust(9, " "), "|", str(conf_mat[0][1]).rjust(9, " "), "|")
	print("targets", "|", "  sat", "|", str(conf_mat[1][0]).rjust(9, " "), "|", str(conf_mat[1][1]).rjust(9, " "), "|")
	print("        +-------+-----------+-----------+")
	print("       ", "|", "     ", "|", "  unsat  ", "|", "   sat   ", "|")
	print("        +-------+-----------+-----------+")
	print("                      preds", "\n", sep="")


def evaluate_model_on_dataset(model, dataset, tokenizer):
	print(f"\nevaluating on {dataset.stage} dataset (len is {len(dataset)})\n")

	dataset = datasets.Dataset.from_generator(dataset.get_iterator)
	dataset = dataset.map(lambda s: tokenizer(s["text"], truncation=True, padding="max_length"), batched=True)
	dataset.set_format("torch")

	device = model.device
	print("\n", device, "\n", sep="")

	dataloader = DataLoader(dataset, batch_size=DEVICE_VAL_BATCH_SIZE)

	metric_evaluators = { 
		metric_name: evaluate.load(metric_name) \
			for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]
	}

	all_outputs, all_targets = [], []
	for batch in tqdm(dataloader):
		batch = { k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items() }

		with torch.no_grad():
			out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

		scores = torch.softmax(out.logits, dim=-1)[:, 1]
		predictions = torch.argmax(out.logits, dim=-1)
		
		for metric_name, metric in metric_evaluators.items():
			if metric_name.endswith("roc_auc"):
				metric.add_batch(references=batch["label"], prediction_scores=scores)
			else:
				metric.add_batch(references=batch["label"], predictions=predictions)

		all_outputs.append(predictions.detach())
		all_targets.append(batch["label"].detach())

	metrics_dict = dict()
	for metric_name, metric in metric_evaluators.items():
		metrics_dict[metric_name] = list(metric.compute().values())[0]

	print()
	for metric_name, metric_value in metrics_dict.items():
		print(metric_name.rjust(11, " "), ":", metric_value)

	all_outputs = torch.flatten(torch.cat(all_outputs))
	all_targets = torch.flatten(torch.cat(all_targets))

	conf_mat = binary_confusion_matrix(all_outputs, all_targets, normalize="all").cpu().numpy()
	print_confusion_matrix(conf_mat)

	conf_mat = binary_confusion_matrix(all_outputs, all_targets, normalize=None).cpu().numpy()
	print_confusion_matrix(conf_mat)


def run_gpt2_experiment(model_name, dataset_labels, train_model, train_tokenizer):
	if train_tokenizer and not train_model:
		raise Error("can't retrain tokenizer without retraining model")

	print(f"\ntraining {model_name} on {dataset_labels}, train_model={train_model}, train_tokenizer={train_tokenizer}\n")

	train_ds = get_dataset("train", dataset_labels)
	val_ds = get_dataset("val", dataset_labels)
	test_ds = get_dataset("test", dataset_labels)

	tokenizer = AutoTokenizer.from_pretrained("gpt2")

	if train_tokenizer:
		tokenizer.normalizer = nSequence([
			Replace("(", " ( "),
			Replace(")", " ) "),
			Replace("!", " ! "),
			Replace("#", " # "),
			Replace("@", " @ "),
		])
		tokenizer.pre_tokenizer = pSequence([
			Punctuation(),
			Whitespace(),
			ByteLevel()
		])
		tokenizer.train_new_from_iterator(train_ds.get_iterator_for_tokenizer(), VOCAB_SIZE)

	tokenizer.pad_token = tokenizer.eos_token

	train_ds.setup(tokenizer, FORMULA_MAX_LENGTH_IN_TOKENS)
	val_ds.setup(tokenizer, FORMULA_MAX_LENGTH_IN_TOKENS)
	test_ds.setup(tokenizer, FORMULA_MAX_LENGTH_IN_TOKENS)

	if train_model:
		config = transformers.GPT2Config(
			vocab_size=VOCAB_SIZE,
			n_positions=FORMULA_MAX_LENGTH_IN_TOKENS,
			pad_token_id=tokenizer.eos_token_id,
		)

		model = transformers.GPT2ForSequenceClassification(config)

		training_args = transformers.TrainingArguments(
			output_dir=model_name,
			per_device_train_batch_size=DEVICE_TRAIN_BATCH_SIZE,
			per_device_eval_batch_size=DEVICE_VAL_BATCH_SIZE,
			num_train_epochs=TRAIN_EPOCHS,
			gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
			# learning_rate=2e-4,
			weight_decay=1,
			logging_strategy="steps",
			logging_steps=0.01,
			evaluation_strategy="epoch",
			save_strategy="epoch",
			load_best_model_at_end=True,
			metric_for_best_model="eval_roc_auc",
			greater_is_better=True,
			push_to_hub=False,
			report_to="tensorboard",
			logging_dir=f"{model_name}-logs",
			dataloader_drop_last=True,
			dataloader_num_workers=32,
		)

		trainer = transformers.Trainer(
			model=model,
			args=training_args,
			train_dataset=train_ds,
			eval_dataset=val_ds,
			tokenizer=tokenizer,
			compute_metrics=compute_metrics,
		)

	else:
		model = transformers.AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2)
		model.config.pad_token_id = model.config.eos_token_id

		training_args = transformers.TrainingArguments(
			output_dir=model_name,
			per_device_train_batch_size=DEVICE_TRAIN_BATCH_SIZE,
			per_device_eval_batch_size=DEVICE_VAL_BATCH_SIZE,
			num_train_epochs=FINETUNE_EPOCHS,
			gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
			# learning_rate=2e-4,
			weight_decay=1,
			logging_strategy="steps",
			logging_steps=0.01,
			evaluation_strategy="epoch",
			save_strategy="epoch",
			load_best_model_at_end=True,
			metric_for_best_model="eval_roc_auc",
			greater_is_better=True,
			push_to_hub=False,
			report_to="tensorboard",
			logging_dir=f"{model_name}-logs",
			dataloader_drop_last=True,
			dataloader_num_workers=32,
		)

		trainer = transformers.Trainer(
			model=model,
			args=training_args,
			train_dataset=train_ds,
			eval_dataset=val_ds,
			tokenizer=tokenizer,
			compute_metrics=compute_metrics,
		)

	trainer.train()

	model.save_pretrained(f"{model_name}-best")
	tokenizer.save_pretrained(f"{model_name}-tokenizer")

	evaluate_model_on_dataset(model, val_ds, tokenizer)
	evaluate_model_on_dataset(model, test_ds, tokenizer)


run_gpt2_experiment(
	model_name=sys.argv[1],
	train_model=("model" in sys.argv[2]),
	train_tokenizer=("tokenizer" in sys.argv[2]),
	dataset_labels=sys.argv[3:],
)
