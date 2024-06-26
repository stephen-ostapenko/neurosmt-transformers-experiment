#!/bin/python3

import sys
import os
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from torchmetrics.classification import BinaryAveragePrecision
from torchmetrics.functional.classification import binary_confusion_matrix

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import evaluate
import datasets


FORMULA_MAX_LENGTH_IN_TOKENS = 2048
TOKENIZER_BATCH_SIZE = 2 ** 10

DEVICE_VAL_BATCH_SIZE = os.environ["DEVICE_VAL_BATCH_SIZE"]
if DEVICE_VAL_BATCH_SIZE is None:
	DEVICE_VAL_BATCH_SIZE = 2
else:
	DEVICE_VAL_BATCH_SIZE = int(DEVICE_VAL_BATCH_SIZE)

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
					print("w: ignoring formula because of it has too many tokens")
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


def get_file_paths(stages, dataset_labels):
	file_paths = []
	
	for ds in dataset_labels:
		for stage in stages:
			with open(f"dataset/{ds}/__meta/{stage}", "r") as f:
				file_paths += [f"dataset/{ds}/{file_path.strip()}" for file_path in f.readlines()]

	return file_paths


def get_dataset(stages, dataset_labels):
	file_paths = get_file_paths(stages, dataset_labels)
	if "SHRINK" in os.environ:
		file_paths = file_paths[:int(os.environ["SHRINK"])]

	formulas, labels = [], []
	for file_path in tqdm(file_paths, "reading files with formulas"):
		with open(file_path, "r") as f:
			formula = f.read()

		label = 0 if file_path.endswith("-unsat") else 1

		formulas.append(formula)
		labels.append(label)

	return FormulaDataset(formulas, labels, "/".join(stages))


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
	if len(dataset) == 0:
		print("skipping")
		return

	dataset = datasets.Dataset.from_generator(dataset.get_iterator)
	dataset = dataset.map(
		lambda s: tokenizer(s["text"], padding="max_length", max_length=FORMULA_MAX_LENGTH_IN_TOKENS),
		batched=True
	)
	dataset.set_format("torch")

	device = model.device
	if str(device) == "cpu" and torch.cuda.is_available():
		device = torch.device("cuda:0")

	print("\n", device, "\n", sep="")
	model = model.to(device)

	dataloader = DataLoader(dataset, batch_size=DEVICE_VAL_BATCH_SIZE)

	metric_evaluators = { 
		metric_name: evaluate.load(metric_name) \
			for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]
	}
	metric_evaluators["avg_precision"] = BinaryAveragePrecision()

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
			elif metric_name.endswith("avg_precision"):
				metric.update(target=batch["label"], preds=scores)
			else:
				metric.add_batch(references=batch["label"], predictions=predictions)

		all_outputs.append(predictions.detach())
		all_targets.append(batch["label"].detach())

	metrics_dict = dict()
	for metric_name, metric in metric_evaluators.items():
		if metric_name.endswith("avg_precision"):
			metrics_dict[metric_name] = metric.compute().item()
		else:
			metrics_dict[metric_name] = list(metric.compute().values())[0]

	print()
	for metric_name, metric_value in metrics_dict.items():
		print(metric_name.rjust(15, " "), ":", metric_value)

	all_outputs = torch.flatten(torch.cat(all_outputs))
	all_targets = torch.flatten(torch.cat(all_targets))

	conf_mat = binary_confusion_matrix(all_outputs, all_targets, normalize="all").cpu().numpy()
	print_confusion_matrix(conf_mat)

	conf_mat = binary_confusion_matrix(all_outputs, all_targets, normalize=None).cpu().numpy()
	print_confusion_matrix(conf_mat)


def run_roberta_evaluation(model_name, dataset_labels, merge_stages):
	print(f"\nevaluating {model_name} on {dataset_labels}\n")

	tokenizer = AutoTokenizer.from_pretrained(f"{model_name}-tokenizer", local_files_only=True)
	model = AutoModelForSequenceClassification.from_pretrained(f"{model_name}-best", local_files_only=True)

	if merge_stages:
		ds = get_dataset(["train", "val", "test"], dataset_labels)

		ds.setup(tokenizer, FORMULA_MAX_LENGTH_IN_TOKENS)

		evaluate_model_on_dataset(model, ds, tokenizer)

	else:
		train_ds = get_dataset(["train"], dataset_labels)
		val_ds = get_dataset(["val"], dataset_labels)
		test_ds = get_dataset(["test"], dataset_labels)

		train_ds.setup(tokenizer, FORMULA_MAX_LENGTH_IN_TOKENS)
		val_ds.setup(tokenizer, FORMULA_MAX_LENGTH_IN_TOKENS)
		test_ds.setup(tokenizer, FORMULA_MAX_LENGTH_IN_TOKENS)

		evaluate_model_on_dataset(model, train_ds, tokenizer)
		evaluate_model_on_dataset(model, val_ds, tokenizer)
		evaluate_model_on_dataset(model, test_ds, tokenizer)


def get_args():
	parser = ArgumentParser(description="validation script")
	parser.add_argument("--name", required=True)
	parser.add_argument("--ds", required=True, nargs="+")
	parser.add_argument("--merge_stages", action="store_true")

	args = parser.parse_args()
	print("args:")
	for arg in vars(args):
		print(arg, "=", getattr(args, arg))

	print()

	return args


args = get_args()

run_roberta_evaluation(
	model_name=args.name,
	dataset_labels=args.ds,
	merge_stages=args.merge_stages,
)
