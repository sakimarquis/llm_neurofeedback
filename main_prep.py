"""
Given a model, prepare the scores for later experiments.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import load
import argparse
from analysis.process_hidden import get_hiddens, train_classify_hiddens, eval_classify_hiddens, process_hiddens, \
    get_tags
from data.load import load_dataset
from utils import seed_everything, load_lm, load_exp_cfg, safe_dump
import gc


def save_NF_data(model, tokenizer, dataset, dataset_label_name, tags, cfg, save_dir):
    data_dict = {"train": {}, "test": {}}
    for partition in ['train', 'test']:
        data, labels = dataset[partition]['data'], dataset[partition][dataset_label_name]
        if len(data) == 0:
            data_dict[partition]["X"] = np.array([])
            data_dict[partition]["y"] = np.array([])
        else:
            logits, hiddens = get_hiddens(model, tokenizer, data, batch_size=cfg.batch_size)
            X, y = process_hiddens(hiddens, tokenizer, data, tags, cfg.process_hidden_method, labels)
            data_dict[partition]["X"] = X
            data_dict[partition]["y"] = y
    safe_dump(data_dict, save_dir / f"hidden_{cfg.process_hidden_method}_data_Xy.pkl")
    print("Data prepared and saved.")


def train_classifier(cfg, save_dir, file_name):
    data_dict = load(save_dir / f"hidden_{cfg.process_hidden_method}_data_Xy.pkl")
    all_classifiers, all_train_accuracies = train_classify_hiddens(
        data_dict['train']["X"], data_dict['train']["y"], cfg.clf, cfg.normalize, pc_number=cfg.pc_number)

    safe_dump(all_classifiers, save_dir / f"hidden_{cfg.process_hidden_method}_classifiers_{file_name}.pkl")
    print("Classifiers trained and saved.")

    if cfg.eval_clf:
        all_classifiers = load(save_dir / f"hidden_{cfg.process_hidden_method}_classifiers_{file_name}.pkl")
        train_X, train_y = data_dict["train"]["X"], data_dict["train"]["y"]
        test_X, test_y = data_dict["test"]["X"], data_dict["test"]["y"]
        all_train_accuracies = eval_classify_hiddens(train_X, train_y, all_classifiers, return_type='accuracy')
        all_test_accuracies = eval_classify_hiddens(test_X, test_y, all_classifiers, return_type='accuracy')
        plt.figure(figsize=(3, 3))
        plt.plot(list(all_train_accuracies.keys()), list(all_train_accuracies.values()), label="Train")
        plt.plot(list(all_test_accuracies.keys()), list(all_test_accuracies.values()), label="Test")
        plt.axhline(y=0.5, color='r', linestyle='--', label="Chance level")  # chance level
        plt.xlabel("Layer")
        plt.ylabel(f"{file_name} accuracy")
        plt.title(f"Hidden {cfg.process_hidden_method} Classifier")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"hidden_{cfg.process_hidden_method}_layer_accuracies_{file_name}.pdf", bbox_inches='tight')
        plt.close()

        all_test_scores = eval_classify_hiddens(test_X, test_y, all_classifiers, return_type='score')
        plt.figure(figsize=(12, 12))
        n_layers = len(list(all_classifiers.keys()))
        n_row = int(n_layers ** 0.5) + 1
        for layer in all_classifiers.keys():
            plt.subplot(n_row, n_row, layer + 1)
            score = all_test_scores[layer]
            plt.hist(score, bins=20)
            plt.title(layer)
        plt.tight_layout()
        plt.savefig(save_dir / f"hidden_{cfg.process_hidden_method}_test_scores_{file_name}.png")
        plt.close()


def generate_example_scores(dataset, model, tokenizer, tags, cfg, save_dir, file_name, data_part='test'):
    seed_everything(42)
    all_classifiers = load(save_dir / f"hidden_{cfg.process_hidden_method}_classifiers_{file_name}.pkl")
    examples_csv = pd.DataFrame({
        'user_prompt': dataset[data_part]['user_data'],
        'assistant_response': dataset[data_part]['assistant_data'],
        'full_prompt': dataset[data_part]['data'],
        # 'full_prompt': dataset[data_part]['data'],
    })
    full_prompts = examples_csv['full_prompt'].tolist()
    logits, hiddens = get_hiddens(model, tokenizer, full_prompts, batch_size=cfg.batch_size)
    processed_hiddens, _ = process_hiddens(hiddens, tokenizer, full_prompts, tags, method=cfg.process_hidden_method)
    scores = eval_classify_hiddens(processed_hiddens, None, all_classifiers,
                                   return_type='score')  # scores[layer][seq_idx]
    examples_scores = pd.concat([examples_csv, pd.DataFrame(scores)], axis=1)

    plt.figure(figsize=(12, 12))
    n_layers = len(list(all_classifiers.keys()))
    n_row = int(n_layers ** 0.5) + 1
    for layer in all_classifiers.keys():
        plt.subplot(n_row, n_row, layer + 1)
        score = scores[layer]
        plt.hist(score, bins=20)
        plt.title(layer)
    plt.tight_layout()
    plt.savefig(save_dir / f"hidden_{cfg.process_hidden_method}_{file_name}_generate_example_scores.png")
    plt.close()
    return examples_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare scores")
    parser.add_argument("--model", type=str, default="llama3.1_8b")
    parser.add_argument("--dataset", type=str, default="true_false")  # commonsense, true_false
    # python main_prep.py --model llama3.1_8b --dataset commonsense
    args = parser.parse_args()
    cfg = load_exp_cfg(args.model)
    cfg.eval_clf = True  # no need to eval clf because we are only obtaining scores
    torch.set_grad_enabled(False)

    tags = get_tags(cfg.model_name)
    model, tokenizer = load_lm(cfg.model_name, use_transformer_lens=cfg.use_transformer_lens, padding_side=cfg.padding_side)
    dataset, dataset_name, dataset_label_name = load_dataset(args.dataset, tags, tokenizer)

    save_dir = Path("results") / cfg.model_name.replace("/", "_") / dataset_name
    os.makedirs(save_dir, exist_ok=True)
    save_NF_data(model, tokenizer, dataset, dataset_label_name, tags, cfg, save_dir)

    cfg.clf = "lr"
    file_name = cfg.clf
    train_classifier(cfg, save_dir, file_name)
    examples_scores = generate_example_scores(dataset, model, tokenizer, tags, cfg, save_dir, file_name, data_part='test')
    safe_dump(examples_scores, save_dir / f"hidden_{cfg.process_hidden_method}_{file_name}_example_scores.pkl")

    cfg.clf = "pcascore"
    for pc_number in cfg.all_pc_exp:
        cfg.pc_number = pc_number
        cfg.clf_name = f"pc{pc_number}"
        file_name = f'{cfg.clf}_{cfg.clf_name}'
        train_classifier(cfg, save_dir, file_name)
        examples_scores = generate_example_scores(dataset, model, tokenizer, tags, cfg, save_dir, file_name, data_part='test')
        safe_dump(examples_scores, save_dir / f"hidden_{cfg.process_hidden_method}_{file_name}_example_scores.pkl")
        del examples_scores
        gc.collect()
