"""Given n ICL examples (sentences, neurofeedback scores), control the model to generate desired scores for new examples"""

import os
from pathlib import Path
import numpy as np
from joblib import load
import argparse
from neurofeedback import imitate_score_by_examples
from utils import seed_everything, load_lm, load_exp_cfg, load_labeler, sample_layers

from tqdm import trange


# VERBOSE = False
# if not VERBOSE:
#     tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Setup")
    parser.add_argument("--model", type=str, default="llama3_3b", help="Configuration file: the model that generates the scores")
    parser.add_argument("--dataset", type=str, default="commonsense")  # commonsense, true_false, sycophancy
    parser.add_argument("--clf", type=str, default="default")  # default classifier in loaded cfg
    parser.add_argument("--pc", type=int, default=1)
    parser.add_argument("--scale", type=int, default=2, help="Score scale: 2 for binary, >2 for Likert scale.")
    parser.add_argument("--n_train", type=int, default=-1, help="Number of training examples to use. -1 means all.")
    args = parser.parse_args()
    cfg = load_exp_cfg(args.model, pc_number=args.pc, clf=args.clf)

    save_dir = Path("results") / cfg.model_name.replace("/", "_") / args.dataset
    if cfg.clf == "lr":
        f_name = clf_name = cfg.clf
    else:
        clf_name = f'{cfg.clf}_pc1'
        f_name = f'{cfg.clf}_pc{cfg.pc_number}'
    examples_scores = load(save_dir / f"hidden_{cfg.process_hidden_method}_{f_name}_example_scores.pkl")
    all_classifiers = load(save_dir / f"hidden_{cfg.process_hidden_method}_classifiers_{clf_name}.pkl")
    selected_layers = sample_layers(list(all_classifiers.keys()), cfg.layers)

    if args.n_train == -1:
        n_train_examples_list = cfg.n_icl_examples_control
    else:
        n_train_examples_list = [args.n_train]

    model, tokenizer = load_lm(cfg.model_name)
    labeler, exp_save_dir = load_labeler(args.scale, cfg.quantile_transform, save_dir)
    total_examples = n_train_examples_list[-1] + 1  # 1 for the test example

    exp_examples_idx = []
    for imit_exp in trange(cfg.exp_id_start, cfg.exp_id_end, desc="Control experiment"):
        seed_everything(42 + imit_exp)
        examples_scores_exp = examples_scores.sample(total_examples, random_state=42 + imit_exp)
        exp_examples_idx.append(examples_scores_exp['sentences'].index.to_list())

        for layer in selected_layers:
            labeler.fit(examples_scores[layer].to_numpy())  # fit labeler on all data
            file_name = f"control_hidden_{cfg.process_hidden_method}_clf_{f_name}_layer{layer}_exp{imit_exp}.pkl"
            if os.path.exists(exp_save_dir / file_name):
                print(f"Layer {layer}, experiment {imit_exp} already completed.")
                continue
            imitate_score_by_examples(model, tokenizer, all_classifiers, examples_scores_exp, layer, labeler,
                                      n_train_examples_list, cfg.process_hidden_method, exp_save_dir / file_name, cfg.pc_number)
        print(f"Experiment {imit_exp} completed and saved.")

    np.savez_compressed(exp_save_dir / f"control_examples_idx.npz", examples_idx=np.array(exp_examples_idx))  # shape: (n_experiments, total_examples)
