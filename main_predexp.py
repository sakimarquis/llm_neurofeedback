"""
Given a model, prepare the scores for later experiments.
"""

import os
from pathlib import Path
import pandas as pd
import torch
from tqdm import trange

from joblib import load
import argparse
from analysis.process_hidden import get_tags
from neurofeedback import predict_score_by_examples
from utils import seed_everything, load_lm, load_exp_cfg, safe_dump
import gc


if __name__ == "__main__":
    VERBOSE = False
    parser = argparse.ArgumentParser(description="Experiment Setup")
    parser.add_argument("--pred_exp_start", type=int, default=0, help="Starting experiment number")
    parser.add_argument("--pred_exp_end", type=int, default=100, help="Ending experiment number (exclusive)")
    parser.add_argument("--config_s", type=str, default="qwen2.5_7b", help="Configuration file: the model that generates the scores")
    parser.add_argument("--config_e", type=str, default="llama3_1b", help="Configuration file: the model that runs the experiments")
    parser.add_argument("--dataset", type=str, default="sycophancy")  # commonsense, true_false, sycophancy
    parser.add_argument("--clf", type=str, default="default")  # default classifier in loaded cfg
    parser.add_argument("--pc", type=int, default=1)
    args = parser.parse_args()
    pred_exp_start = args.pred_exp_start
    pred_exp_end = args.pred_exp_end
    if pred_exp_end == -1:
        pred_exp_end = pred_exp_start + 1
    cfg_s = load_exp_cfg(args.config_s, pc_number=args.pc, clf=args.clf)  # pc of the model's hiddens
    cfg_e = load_exp_cfg(args.config_e)
    torch.set_grad_enabled(False)
    seed_everything(42)

    dataset_name, label_name = args.dataset, "labels"
    save_dir = Path("results") / cfg_s.model_name.replace("/", "_") / dataset_name
    f_name = cfg_s.clf if cfg_s.clf == "lr" else f'{cfg_s.clf}_{cfg_s.clf_name}'
    examples_scores = load(save_dir / f"hidden_{cfg_s.process_hidden_method}_{f_name}_example_scores.pkl")
    all_classifiers = load(save_dir / f"hidden_{cfg_s.process_hidden_method}_classifiers_{f_name}.pkl")
    all_layers = list(all_classifiers.keys())

    save_dir = Path("results") / (cfg_s.model_name.replace("/", "_") + '-'+ cfg_e.model_name.replace("/", "_")) / dataset_name
    os.makedirs(save_dir, exist_ok=True)
    tags = get_tags(cfg_e.model_name)
    model, tokenizer = load_lm(cfg_e.model_name, use_transformer_lens=cfg_e.use_transformer_lens, padding_side=cfg_e.padding_side)

    for pred_exp in trange(pred_exp_start, pred_exp_end, desc="Experiment"):
        print(f"Starting from scratch for experiment {pred_exp}.")
        file_name = f"predict_score_by_examples_hidden_{cfg_s.process_hidden_method}_clf_{f_name}_exp{pred_exp}.pkl"
        if os.path.exists(save_dir / file_name):
            print(f"Experiment {pred_exp} already completed.")
            continue
        seed_everything(42 + pred_exp)
        examples_scores_exp = examples_scores.sample(len(examples_scores))
        est_score_dt = predict_score_by_examples(model, tokenizer, examples_scores_exp, binary_score=True,
                                                 all_layers=all_layers, verbose=VERBOSE)
        est_score_dt['experiment'] = [pred_exp] * len(est_score_dt['prompt'])
        est_score_dt = pd.DataFrame(est_score_dt)
        safe_dump(est_score_dt, save_dir / file_name)
        print(f"Experiment {pred_exp} completed and saved.")
        del est_score_dt
        gc.collect()
