"""
Given a model, prepare the scores for later experiments.
"""

import os
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from joblib import load
import argparse
from tqdm import tqdm, trange
from functools import partialmethod

VERBOSE = False
if not VERBOSE:
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from analysis.process_hidden import get_tags
from neurofeedback import imitate_score_by_examples
from utils import seed_everything, load_lm, load_exp_cfg, safe_dump, Binarizer, EvenLikertBinner


if __name__ == "__main__":
    # VERBOSE = False
    parser = argparse.ArgumentParser(description="Experiment Setup")
    parser.add_argument("--imit_exp_start", type=int, default=0, help="Starting experiment number")
    parser.add_argument("--imit_exp_end", type=int, default=100, help="Ending experiment number (exclusive)")
    parser.add_argument("--model", type=str, default="llama3_1b", help="Configuration file: the model that generates the scores")
    parser.add_argument("--layer", type=int, default=0, help="Layer to use for imitation")
    parser.add_argument("--dataset", type=str, default="sycophancy")  # commonsense, true_false, sycophancy
    parser.add_argument("--clf", type=str, default="default")  # default classifier in loaded cfg
    parser.add_argument("--pc", type=int, default=1)
    parser.add_argument("--n_train", type=int, default=-1, help="Number of training examples to use. -1 means all.")
    args = parser.parse_args()
    cfg = load_exp_cfg(args.model, pc_number=args.pc, clf=args.clf)
    imit_exp_start = args.imit_exp_start
    imit_exp_end = args.imit_exp_end
    if imit_exp_end == -1:
        imit_exp_end = imit_exp_start + 1
    torch.set_grad_enabled(False)
    seed_everything(42)

    dataset_name, label_name = args.dataset, "labels"
    save_dir = Path("results") / cfg.model_name.replace("/", "_") / dataset_name
    f_name = cfg.clf if cfg.clf == "lr" else f'{cfg.clf}_{cfg.clf_name}'
    examples_scores = load(save_dir / f"hidden_{cfg.process_hidden_method}_{f_name}_example_scores.pkl")
    all_classifiers = load(save_dir / f"hidden_{cfg.process_hidden_method}_classifiers_{f_name}.pkl")

    selected_layers = list(all_classifiers.keys())
    if args.layer == -1:  # only use the middle layer
        selected_layers = [int(np.median(selected_layers))]
    elif args.layer == 0:  # sample the layers
        selected_layers = np.percentile(selected_layers, [0, 25, 50, 75, 100], interpolation='lower')
    else:
        selected_layers = [args.layer]

    if args.n_train == -1:
        n_train_examples_list = cfg.n_train_examples
    else:
        n_train_examples_list = [args.n_train]

    tags = get_tags(cfg.model_name)
    model, tokenizer = load_lm(cfg.model_name, use_transformer_lens=cfg.use_transformer_lens, padding_side=cfg.padding_side)
    binary_score = True

    if binary_score:
        labeler = Binarizer()
        exp_save_dir = save_dir
    else:
        labeler = EvenLikertBinner(6)
        exp_save_dir = save_dir = Path("results") / cfg.model_name.replace("/", "_") / f'{dataset_name}_6_points'
    os.makedirs(exp_save_dir, exist_ok=True)

    for n_train_examples in n_train_examples_list:
        total_examples = n_train_examples + 1  # 1 for the test example
        print("n_train_examples:", n_train_examples, "total_examples:", total_examples)

        for layer in selected_layers:
            labeler.fit(examples_scores[layer].to_numpy())

            for imitate_mode in ['active', 'inactive']:
                for imit_exp in trange(imit_exp_start, imit_exp_end, desc="Experiment"):
                    print(f"Starting from scratch for experiment {imit_exp}.")
                    file_name = f"imitate_score_by_examples_hidden_{cfg.process_hidden_method}_clf_{f_name}_layer{layer}_ntrain{n_train_examples}_mode{imitate_mode}_exp{imit_exp}.pkl"
                    if os.path.exists(exp_save_dir / file_name):
                        print(f"Experiment {imit_exp} already completed.")
                        continue
                    seed_everything(42 + imit_exp)
                    examples_scores_exp = examples_scores.sample(total_examples)
                    est_score_dt = imitate_score_by_examples(model, tokenizer,  tags, all_classifiers, examples_scores_exp, layer, labeler,
                                                             binary_score=binary_score, process_hidden_method=cfg.process_hidden_method,
                                                             imitate_mode=imitate_mode, verbose=VERBOSE)
                    est_score_dt['experiment'] = [imit_exp] * len(est_score_dt['prompt'])
                    est_score_dt = pd.DataFrame(est_score_dt)
                    safe_dump(est_score_dt, exp_save_dir / file_name)
                    print(f"Experiment {imit_exp} completed and saved.")
                    del est_score_dt
                    gc.collect()
