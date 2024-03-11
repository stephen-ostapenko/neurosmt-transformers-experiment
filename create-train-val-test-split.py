#!/bin/python3

import os
from argparse import ArgumentParser

from tqdm import tqdm, trange
import numpy as np

from tokenizers.normalizers import Replace, Sequence as nSequence
from tokenizers.pre_tokenizers import ByteLevel, Punctuation, Sequence as pSequence, Whitespace

from transformers import AutoTokenizer


METADATA_PATH = "__meta"
VOCAB_SIZE = 2 ** 17
FORMULA_MAX_LENGTH_IN_CHARS = 2 ** 19
FORMULA_MAX_LENGTH_IN_TOKENS = 1024
TOKENIZER_BATCH_SIZE = 2 ** 10


def get_batched_iterator_for_tokenizer(path_to_dataset_root):
	def get_iterator_for_tokenizer(path_to_dataset_root):
		for root, dirs, files in os.walk(path_to_dataset_root, topdown=True):
			dirs = list(filter(lambda d: not d.startswith(METADATA_PATH), dirs))

			for file_name in files:
				with open(os.path.join(root, file_name), "r") as f:
					formula = f.read()
				
				if len(formula) > FORMULA_MAX_LENGTH_IN_CHARS:
					continue

				yield formula
	
	batch = []
	for formula in get_iterator_for_tokenizer(path_to_dataset_root):
		if len(batch) == TOKENIZER_BATCH_SIZE:
			yield batch
			batch = []

		batch.append(formula)
	
	if len(batch) > 0:
		yield batch
	

def get_trained_tokenizer(path_to_dataset_root):
	tokenizer = AutoTokenizer.from_pretrained("gpt2")
	
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
	tokenizer.train_new_from_iterator(
		get_batched_iterator_for_tokenizer(path_to_dataset_root),
		VOCAB_SIZE
	)
	tokenizer.pad_token = tokenizer.eos_token

	return tokenizer


def get_list_of_suitable_samples(path_to_dataset_root, tokenizer):
	def get_samples_iterator(path_to_dataset_root):
		for root, dirs, files in os.walk(path_to_dataset_root, topdown=True):
			dirs = list(filter(lambda d: not d.startswith(METADATA_PATH), dirs))

			for file_name in files:
				cur_path = os.path.join(root, file_name)
				with open(cur_path, "r") as f:
					formula = f.read()
				
				if len(formula) > FORMULA_MAX_LENGTH_IN_CHARS:
					continue

				yield formula, os.path.relpath(cur_path, path_to_dataset_root)
	
	def get_batched_samples_iterator(path_to_dataset_root):
		batch = []
		for sample in get_samples_iterator(path_to_dataset_root):
			if len(batch) == TOKENIZER_BATCH_SIZE:
				yield batch
				batch = []

			batch.append(sample)
		
		if len(batch) > 0:
			yield batch

	suitable_paths = []
	for batch in tqdm(get_batched_samples_iterator(path_to_dataset_root)):
		formulas, paths = zip(*batch)
		formulas = tokenizer(formulas)
		
		for formula, path in zip(formulas["input_ids"], paths):
			if len(formula) <= FORMULA_MAX_LENGTH_IN_TOKENS:
				suitable_paths.append(path)
	
	print(f"got {len(suitable_paths)} correct samples overall")
	return suitable_paths


def calc_group_weights(path_to_dataset_root, tokenizer, list_of_suitable_samples):
	groups = dict()
	for path_to_sample in list_of_suitable_samples:
		group = path_to_sample.split("/")[0].strip()
		if group not in groups:
			groups[group] = 0

		groups[group] += 1

	return list(groups.items())


def align_sat_unsat_sizes_with_upsamping(sat_data: list[str], unsat_data: list[str]) -> tuple[list[str], list[str]]:
	sat_cnt = len(sat_data)
	unsat_cnt = len(unsat_data)

	sat_indices = list(range(sat_cnt))
	unsat_indices = list(range(unsat_cnt))

	if sat_cnt < unsat_cnt:
		sat_indices += list(np.random.choice(np.array(sat_indices), unsat_cnt - sat_cnt, replace=True))
	elif sat_cnt > unsat_cnt:
		unsat_indices += list(np.random.choice(np.array(unsat_indices), sat_cnt - unsat_cnt, replace=True))

	return (
		list(np.array(sat_data, dtype=object)[sat_indices]),
		list(np.array(unsat_data, dtype=object)[unsat_indices])
	)


def align_sat_unsat_sizes_with_downsamping(sat_data: list[str], unsat_data: list[str]) -> tuple[list[str], list[str]]:
	sat_cnt = len(sat_data)
	unsat_cnt = len(unsat_data)

	sat_indices = list(range(sat_cnt))
	unsat_indices = list(range(unsat_cnt))

	if sat_cnt > unsat_cnt:
		sat_indices = np.random.choice(np.array(sat_indices), unsat_cnt, replace=False)
	elif sat_cnt < unsat_cnt:
		unsat_indices = np.random.choice(np.array(unsat_indices), sat_cnt, replace=False)

	return (
		list(np.array(sat_data, dtype=object)[sat_indices]),
		list(np.array(unsat_data, dtype=object)[unsat_indices])
	)


def align_sat_unsat_sizes(sat_data: list[str], unsat_data: list[str], mode: str) -> tuple[list[str], list[str]]:
	if mode == "none":
		return sat_data, unsat_data
	elif mode == "upsample":
		return align_sat_unsat_sizes_with_upsamping(sat_data, unsat_data)
	elif mode == "downsample":
		return align_sat_unsat_sizes_with_downsamping(sat_data, unsat_data)
	else:
		raise Exception(f"unknown sampling mode {mode}")


def grouped_random_split(
	path_to_dataset_root, tokenizer,
	val_qty, test_qty,
	align_train_mode, align_val_mode, align_test_mode
):
	list_of_suitable_samples = get_list_of_suitable_samples(path_to_dataset_root, tokenizer)
	groups = calc_group_weights(path_to_dataset_root, tokenizer, list_of_suitable_samples)

	def pick_best_split(groups):
		attempts = 1_000_000

		groups_cnt = len(groups)
		samples_cnt = sum(g[1] for g in groups)

		need_val = int(samples_cnt * val_qty)
		need_test = int(samples_cnt * test_qty)
		need_train = samples_cnt - need_val - need_test

		print("picking best split with existing groups")
		print(f"need: {need_train} (train) | {need_val} (val) | {need_test} (test)")
		print(flush=True)

		best = None

		for _ in trange(attempts):
			probs = (np.array([need_train, need_val, need_test]) / samples_cnt + \
				     np.array([1, 1, 1]) / 3) / 2
			
			cur_split = np.random.choice(
				range(3), size=groups_cnt,
				p=probs
			)

			train_size = sum(groups[i][1] for i in range(groups_cnt) if cur_split[i] == 0)
			val_size = sum(groups[i][1] for i in range(groups_cnt) if cur_split[i] == 1)
			test_size = sum(groups[i][1] for i in range(groups_cnt) if cur_split[i] == 2)

			cur_error = (train_size - need_train) ** 2 + (val_size - need_val) ** 2 + (test_size - need_test) ** 2

			if best is None or best[0] > cur_error:
				best = (cur_error, cur_split)

		return best[1]

	split = pick_best_split(groups)
	split_by_group = dict()
	for i, (group, weight) in enumerate(groups):
		split_by_group[group] = split[i]

	train_data, val_data, test_data = [], [], []
	for path_to_sample in list_of_suitable_samples:
		group = path_to_sample.split("/")[0].strip()
		
		if split_by_group[group] == 0:
			train_data.append(path_to_sample)
		elif split_by_group[group] == 1:
			val_data.append(path_to_sample)
		elif split_by_group[group] == 2:
			test_data.append(path_to_sample)

	def split_data_to_sat_unsat(data):
		sat_data = list(filter(lambda path: path.endswith("-sat"), data))
		unsat_data = list(filter(lambda path: path.endswith("-unsat"), data))

		return sat_data, unsat_data

	def align_data(data, mode):
		sat_data, unsat_data = split_data_to_sat_unsat(data)
		sat_data, unsat_data = align_sat_unsat_sizes(sat_data, unsat_data, mode)

		return sat_data + unsat_data

	if align_train_mode != "none":
		train_data = align_data(train_data, align_train_mode)

	if align_val_mode != "none":
		val_data = align_data(val_data, align_val_mode)

	if align_test_mode != "none":
		test_data = align_data(test_data, align_test_mode)

	np.random.shuffle(train_data)
	np.random.shuffle(val_data)
	np.random.shuffle(test_data)

	return train_data, val_data, test_data


def create_split(
	path_to_dataset_root, tokenizer,
	val_qty, test_qty,
	align_train_mode, align_val_mode, align_test_mode
):
	train_data, val_data, test_data = grouped_random_split(
		path_to_dataset_root, tokenizer,
		val_qty, test_qty,
		align_train_mode, align_val_mode, align_test_mode
	)

	print("\nstats:", flush=True)
	print(f"train: {len(train_data)}")
	print(f"val:   {len(val_data)}")
	print(f"test:  {len(test_data)}")
	print(flush=True)

	meta_path = os.path.join(path_to_dataset_root, METADATA_PATH)
	os.makedirs(meta_path, exist_ok=True)

	with open(os.path.join(meta_path, "train"), "w") as f:
		f.write("\n".join(train_data))

		if len(train_data):
			f.write("\n")

	with open(os.path.join(meta_path, "val"), "w") as f:
		f.write("\n".join(val_data))

		if len(val_data):
			f.write("\n")

	with open(os.path.join(meta_path, "test"), "w") as f:
		f.write("\n".join(test_data))

		if len(test_data):
			f.write("\n")


def get_args():
	parser = ArgumentParser(description="train/val/test splitting script")

	parser.add_argument("--ds", required=True)

	parser.add_argument("--val_qty", type=float, default=0.15)
	parser.add_argument("--test_qty", type=float, default=0.1)

	parser.add_argument("--align_train", choices=["none", "upsample", "downsample"], default="none")
	parser.add_argument("--align_val", choices=["none", "upsample", "downsample"], default="none")
	parser.add_argument("--align_test", choices=["none", "upsample", "downsample"], default="none")

	args = parser.parse_args()
	print("args:")
	for arg in vars(args):
		print(arg, "=", getattr(args, arg))

	print()

	return args


if __name__ == "__main__":
	np.random.seed(24)

	args = get_args()

	tokenizer = get_trained_tokenizer(args.ds)

	create_split(
		args.ds, tokenizer,
		args.val_qty, args.test_qty,
		args.align_train, args.align_val, args.align_test,
	)
