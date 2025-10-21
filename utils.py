import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Union, List
import platform
import numpy as np
import pandas as pd
import matplotlib as mpl
import torch
from joblib import dump, load, Parallel, delayed
from ruamel.yaml import YAML
from tqdm import trange
from transformer_lens import utils as utils, HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_yaml(file_path: Union[Path, str]) -> Dict:
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(file_path, 'r', encoding='utf-8') as file:
        yaml_dict = yaml.load(file)
    return yaml_dict


def load_exp_cfg(model_name: str, pc_number: int = 3, clf='default'):
    try:
        cfg = load_yaml(Path('configs') / 'nf_exp1.yml')
    except FileNotFoundError:
        cfg = load_yaml(Path('../configs') / 'nf_exp1.yml')

    if model_name == 'llama3.1_70b' or model_name == 'llama3_70b':
        cfg['model_name'] = 'meta-llama/Llama-3.1-70B-Instruct'
        cfg['n_train_examples'] = [256]  # only do experiments with 256 examples for 70b to save time
    elif model_name == 'llama3.1_8b' or model_name == 'llama3_8b':
        cfg['model_name'] = 'meta-llama/Llama-3.1-8B-Instruct'
    elif model_name == 'llama3.2_1b' or model_name == 'llama3_1b':
        cfg['model_name'] = 'meta-llama/Llama-3.2-1B-Instruct'
    elif model_name == 'llama3.2_3b' or model_name == 'llama3_3b':
        cfg['model_name'] = 'meta-llama/Llama-3.2-3B-Instruct'
    elif model_name == 'qwen2.5_72b':
        cfg['model_name'] = 'Qwen/Qwen2.5-72B-Instruct'
        cfg['n_train_examples'] = [256]
    elif model_name == 'qwen2.5_7b':
        cfg['model_name'] = 'Qwen/Qwen2.5-7B-Instruct'
    elif model_name == 'qwen2.5_1.5b':
        cfg['model_name'] = 'Qwen/Qwen2.5-1.5B-Instruct'
    elif model_name == 'qwen2.5_3b':
        cfg['model_name'] = 'Qwen/Qwen2.5-3B-Instruct'
    else:
        cfg['model_name'] = model_name

    if clf == 'default':
        clf = cfg['clf']
    else:
        cfg['clf'] = clf

    if clf in ['pcascore','pcadiff']:
        cfg['pc_number'] = pc_number
        cfg['clf_name'] = f"pc{pc_number}"
    elif clf in ['lr']:
        cfg['clf_name'] = ''
    else:
        raise ValueError(f"Unknown classifier: {clf}")
    cfg = SimpleNamespace(**cfg)
    return cfg


def load_lm(model_name_or_path, device=None, dtype="float16", use_transformer_lens=True, padding_side="left"):
    if use_transformer_lens:
        device = utils.get_device() if device is None else device
        model = HookedTransformer.from_pretrained(model_name_or_path, device=device, dtype=dtype)
        model.device = device
        model.tokenizer.padding_side = padding_side
        tokenizer = AutoTokenizer.from_pretrained(model.cfg.tokenizer_name, padding_side=padding_side)
        assert_equal_test_prompts(tokenizer, model.to_tokens, device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side, pad_to_multiple_of=8)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print('Note: Padding side is set to', tokenizer.padding_side)
    return model, tokenizer


def get_model_name(model):
    if isinstance(model, HookedTransformer):
        return model.cfg.model_name
    else:
        return model.name_or_path


@torch.inference_mode()
def generate_text(
        model,
        tokenizer,
        prompts,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        keep_new=False,
        batch_size=8,
        verbose=True,
        logits_processor=None,
        skip_special_tokens=False,
):
    # If prompt is a string, convert it to a list to support batching.
    is_single = False
    if isinstance(prompts, str):
        prompts = [prompts]
        is_single = True

    total_length = len(prompts)
    if isinstance(model, HookedTransformer):
        device = model.cfg.device
        tokens = tokenizer(prompts, return_tensors="pt", padding=True)['input_ids'].to(device)
    else:
        device = model.device
        tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    # We track a pointer to where we are in `tokens`, and while there's data left
    # we'll try to generate in batches. If we fail due to OOM, reduce batch_size.
    i = 0
    outputs = []
    generated_texts = []

    with trange(0, total_length, batch_size, desc="Generate all texts", disable=not verbose) as pbar:
        pbar.reset(total=total_length)  # total steps = total_length

        while i < total_length:
            success = False
            while not success:
                try:
                    # Make sure we don't exceed total_length
                    current_batch_end = min(i + batch_size, total_length)

                    if isinstance(model, HookedTransformer):
                        batch = tokenizer(prompts[i:current_batch_end], return_tensors="pt", padding=True)['input_ids'].to(device)
                        output = model.generate(
                            batch,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature,
                            logits_processor=logits_processor
                        )
                    else:
                        batch = tokenizer(prompts[i:current_batch_end], return_tensors="pt", padding=True).to(device)
                        input_length = batch["input_ids"].shape[1]
                        output = model.generate(
                            **batch,
                            max_length=input_length + max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature,
                            logits_processor=logits_processor
                        )
                    success = True
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    # Reduce batch_size by half
                    new_batch_size = max(batch_size // 2, 1)
                    if new_batch_size == batch_size:
                        raise RuntimeError(
                            "Ran out of memory even with batch_size=1. Cannot proceed."
                        )
                    print(f"OutOfMemoryError. Reducing batch_size from {batch_size} to {new_batch_size} and retrying.")
                    batch_size = new_batch_size
                    # Also update the tqdm bar step since we didn't advance yet
                    pbar.total = total_length
                    pbar.n = i
                    pbar.refresh()

            output = list(output.detach().cpu())
            outputs.extend(output)
            for idx, out in enumerate(output):
                if keep_new:
                    # Keep only the newly generated tokens
                    if isinstance(model, HookedTransformer):
                        input_len = batch.shape[1]
                    else:
                        input_len = batch["input_ids"].shape[1]
                    gen_text = tokenizer.decode(out[input_len:], skip_special_tokens=skip_special_tokens)
                else:
                    gen_text = tokenizer.decode(out, skip_special_tokens=skip_special_tokens)
                generated_texts.append(gen_text)

            i = current_batch_end
            pbar.update(current_batch_end - pbar.n)

    assert len(generated_texts) == len(prompts), f"Expected {len(prompts)} generated texts, but got {len(generated_texts)}"
    # If the original input was a single string, return a single string.
    if is_single:
        return generated_texts[0]
    return generated_texts


def assert_equal_test_prompts(tokenizer_hf, tokenizer_tl, device):
    data = [
        'I am a test example.',
        'I am yet another another test example.',
    ]
    tokens_hf = tokenizer_hf(data, return_tensors="pt", padding=True).to(device)['input_ids']
    tokens_tl = tokenizer_tl(data).to(device)
    # print("Test tokenizers:")
    # print(tokens_hf, tokens_tl)
    assert torch.equal(tokens_hf, tokens_tl), "Tokenizers produce different results"


def safe_dump(obj, file):
    try:
        dump(obj, file, compress=('lzma', 3))
    except ValueError:
        dump(obj, file, compress=3)
    except FileNotFoundError:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        dump(obj, file, compress=('lzma', 3))
    except OSError:
        dump(obj, f'{file}_tmp', compress=('lzma', 3))


def parallel_load(file_paths: List[str], n_jobs: int = -1) -> List[object]:
    """Parallel load of multiple joblib files.
    :param file_paths: List of joblib file paths.
    :param n_jobs: Number of parallel workers (-1 = use all cores).
    """
    def safe_load(path):
        try:
            return load(path)
        except Exception as e:
            print(f"[load error] {path}: {e}")
            return None

    if sys.gettrace() is not None:  # if running in a debugger
        results = [safe_load(path) for path in file_paths]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(safe_load)(path) for path in file_paths)
    return [r for r in results if r is not None]


PLOT_PARAMS = {
    "dpi": 300,
    "bbox_inches": 'tight',
    "pad_inches": 0.1,
}


def set_mpl():
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'arial'
    mpl.rcParams['savefig.dpi'] = 600
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False


def load_saved_data(file_name, save_dir, experiment='imitate', start=0, end=100, verbose=False):
    assert experiment in ['imitate', 'predict'], "experiment should be either 'imitate' or 'predict'"
    n_exp = 0
    file_paths = []
    for i_exp in range(start, end):
        name = f"{experiment}_score_by_examples_hidden_last_assistant_to_eos_mean_{file_name}_exp{i_exp}.pkl"
        if os.path.exists(f"{save_dir}/{name}"):
            if verbose:
                print(f"load {save_dir}/{name}")
            file_paths.append(f"{save_dir}/{name}")
            n_exp += 1
        else:
            if verbose:
                print(f"Experiment {save_dir}/{name} not found.")
            continue
    all_scores = parallel_load(file_paths)
    print(f"Loaded {n_exp}/{end - start} experiments for {file_name}")
    all_scores = pd.concat(all_scores, ignore_index=True)
    return all_scores


class Binarizer:
    def __init__(self):
        pass

    def fit(self, x):
        pass

    def transform(self, x):
        return 1 if x >= 0 else 0


class EvenLikertBinner:
    """Splits real-valued scores centered at 0 into n_bins bins, labeling them from 1 to n_bins.
    Negative scores occupy bins 1...(n_bins/2),
    Positive scores occupy bins (n_bins/2+1)...n_bins.
    """
    def __init__(self, n_bins: int):
        """
        :param n_bins: total number of bins (must be even)
        """
        if n_bins % 2 != 0:
            raise ValueError("n_bins must be an even integer.")
        self.n_bins = n_bins
        self.half = n_bins // 2
        self._neg_cutpoints = None
        self._pos_cutpoints = None

    def fit(self, scores: np.ndarray):
        scores = np.asarray(scores)
        neg = scores[scores < 0]
        pos = scores[scores >= 0]

        if len(neg) < self.half or len(pos) < self.half:
            raise ValueError("Not enough negative or non-negative samples to form bins.")

        # Compute quantiles for negative side and force the rightmost edge at 0
        neg_edges = np.quantile(neg, np.linspace(0, 1, self.half + 1))
        neg_edges[-1] = 0.0
        # Exclude first and last to get internal cut points
        self._neg_cutpoints = neg_edges[1:-1]
        # Compute quantiles for positive side and force the leftmost edge at 0
        pos_edges = np.quantile(pos, np.linspace(0, 1, self.half + 1))
        pos_edges[0] = 0.0
        self._pos_cutpoints = pos_edges[1:-1]

    def transform(self, scores: np.ndarray | float | List[float]):
        if self._neg_cutpoints is None or self._pos_cutpoints is None:
            raise ValueError("The binning model has not been fitted yet. Call fit() before transform().")

        if np.isscalar(scores):
            x = float(scores)
            if x < 0:
                return int(np.searchsorted(self._neg_cutpoints, x) + 1)
            else:
                return int(np.searchsorted(self._pos_cutpoints, x) + 1 + self.half)

        arr = np.asarray(scores)
        labels = np.empty_like(arr, dtype=int)

        neg_mask = arr < 0
        if neg_mask.any():
            labels[neg_mask] = np.searchsorted(self._neg_cutpoints, arr[neg_mask]) + 1

        pos_mask = ~neg_mask
        if pos_mask.any():
            labels[pos_mask] = (np.searchsorted(self._pos_cutpoints, arr[pos_mask]) + 1 + self.half)
        return labels


if __name__ == "__main__":
    from time import time
    clf = 'pcascore_pc1'
    layer = 15
    n_train = 256
    mode = 'active'
    save_dir = f"./results/meta-llama_Llama-3.1-8B-Instruct/commonsense"

    time_start = time()
    all_scores = []
    all_paths = []
    for imit_exp in range(100):
        name = f"imitate_score_by_examples_hidden_last_assistant_to_eos_mean_clf_{clf}_layer{layer}_ntrain{n_train}_mode{mode}_exp{imit_exp}.pkl"
        if os.path.exists(f"{save_dir}/{name}"):
            all_paths.append(f"{save_dir}/{name}")
            all_scores.append(load(f"{save_dir}/{name}"))
    time_end = time()
    print(f"Loaded {len(all_scores)} experiments in {time_end - time_start:.2f} seconds.")

    time_start = time()
    all_scores = parallel_load(all_paths)
    time_end = time()
    print(f"Loaded {len(all_scores)} experiments in {time_end - time_start:.2f} seconds.")
