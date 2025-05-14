import os
from joblib import load
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from plot_imit import compute_scores_and_hiddens_diff_lr
from analysis.stats_fn import calc_cohen_d
from utils import load_exp_cfg, set_mpl, PLOT_PARAMS, safe_dump
from configs.settings import SELECTED_LAYERS


def get_all_layers_hiddens(all_scores, layers):
    filter_cond0_label0 = all_scores[(all_scores['imitate_label'] == 0) & (all_scores['flip_shown_label'] == 0)]
    filter_cond1_label1 = all_scores[(all_scores['imitate_label'] == 1) & (all_scores['flip_shown_label'] == 0)]
    filter_cond1_label0 = all_scores[(all_scores['imitate_label'] == 0) & (all_scores['flip_shown_label'] == 1)]
    filter_cond0_label1 = all_scores[(all_scores['imitate_label'] == 1) & (all_scores['flip_shown_label'] == 1)]

    all_layers_hiddens = defaultdict(lambda: defaultdict())
    for layer in layers:
        all_layers_hiddens[layer]['cond0_label0'] = np.array([x[layer] for x in filter_cond0_label0['processed_hiddens'].to_list()]).squeeze()
        all_layers_hiddens[layer]['cond1_label1'] = np.array([x[layer] for x in filter_cond1_label1['processed_hiddens'].to_list()]).squeeze()
        all_layers_hiddens[layer]['cond1_label0'] = np.array([x[layer] for x in filter_cond1_label0['processed_hiddens'].to_list()]).squeeze()
        all_layers_hiddens[layer]['cond0_label1'] = np.array([x[layer] for x in filter_cond0_label1['processed_hiddens'].to_list()]).squeeze()
    return all_layers_hiddens


def compute_all_layers_scores_and_hiddens_diff_lr(classifier, all_layers_hiddens):
    for layer, hiddens in all_layers_hiddens.items():
        hiddens_0 = (hiddens['cond0_label0'] + hiddens['cond1_label1']) / 2
        hiddens_1 = (hiddens['cond1_label0'] + hiddens['cond0_label1']) / 2
        score_0, score_1, all_neural_diff = compute_scores_and_hiddens_diff_lr(
            classifier, hiddens_0, hiddens_1, hiddens['cond0_label0'], hiddens['cond1_label1'])
        all_layers_hiddens[layer]['score_0'] = score_0
        all_layers_hiddens[layer]['score_1'] = score_1
        all_layers_hiddens[layer]['all_neural_diff'] = all_neural_diff
    return all_layers_hiddens


if __name__ == "__main__":
    set_mpl()
    model = "llama3.1_8b"  # "llama3.1_8b" or "qwen2.5_7b" or "llama3.1_70b" or "llama3.2_1b" or "qwen2.5_1.5b" or "llama3.2_3b" or "qwen2.5_3b"
    cfg = load_exp_cfg(model)
    dataset_name, label_name = "commonsense", "labels"
    save_dir = Path("../results") / (cfg.model_name.replace("/", "_")) / dataset_name
    fig_dir = f'{save_dir}/imitation'
    os.makedirs(fig_dir, exist_ok=True)
    n_train_examples = cfg.n_train_examples
    cmap = plt.get_cmap('viridis')
    n_cols = 3
    n_exp = 100

    target_layers = SELECTED_LAYERS[model]  # [15]
    affected_layers = list(range(SELECTED_LAYERS[model][-1] + 1))
    pc_to_control = cfg.all_pc_exp
    pc_to_compare = [i for i in range(512)]
    pc_positions = np.linspace(1, 0, len(pc_to_control))  # we compare control PCs with control PCs
    colors = [cmap(p) for p in pc_positions]

    use_paired_samples = False
    effect_unit = 'SNR' if use_paired_samples else 'd'
    fig_format = 'svg'

    # lr_classifiers = load(save_dir / f"hidden_{cfg.process_hidden_method}_classifiers_lr.pkl")
    #
    # for mode in ['active', 'inactive']:
    #     save_file = f"{fig_dir}/lr_{mode}"
    #     all_score_0 = np.zeros((len(target_layers), len(affected_layers), n_exp, len(n_train_examples)))  # pc and lr
    #     all_score_1 = np.zeros_like(all_score_0)
    #
    #     for i_layer, layer in enumerate(target_layers):
    #         lr_layer_classifier = lr_classifiers[layer]  # project hiddens to the axis defined by target layer
    #
    #         for i_train_example, n_train in enumerate(n_train_examples):
    #             file_name = f'clf_lr_layer{layer}_ntrain{n_train}_mode{mode}'
    #             try:
    #                 all_scores = load_saved_data(file_name, save_dir, start=0, end=0 + n_exp)
    #             except ValueError:  # no objects to concatenate
    #                 print(f"File {file_name} not found.")
    #                 continue
    #
    #             all_layers_hiddens = get_all_layers_hiddens(all_scores, affected_layers)
    #             all_layers_hiddens = compute_all_layers_scores_and_hiddens_diff_lr(lr_layer_classifier, all_layers_hiddens)
    #             for j_layer, affected_layer in enumerate(affected_layers):
    #                 lr_score_0 = all_layers_hiddens[affected_layer]['score_0']
    #                 lr_score_1 = all_layers_hiddens[affected_layer]['score_1']
    #                 all_score_0[i_layer, j_layer, :lr_score_0.shape[0], i_train_example] = lr_score_0
    #                 all_score_1[i_layer, j_layer, :lr_score_1.shape[0], i_train_example] = lr_score_1
    #
    #     safe_dump(all_score_0, f"{save_file}_all_layers_score_0.pkl")
    #     safe_dump(all_score_1, f"{save_file}_all_layers_score_1.pkl")

    i_train = -1
    max_layer = affected_layers[-1]

    for mode in ['active', 'inactive']:
        save_file = f"{fig_dir}/lr_{mode}"
        all_score_0 = load(f"{save_file}_all_layers_score_0.pkl")
        all_score_1 = load(f"{save_file}_all_layers_score_1.pkl")
        control_effect, ci = calc_cohen_d(all_score_0, all_score_1, axis=2)
        upper = control_effect + ci / 1.96
        lower = control_effect - ci / 1.96
        y_min = min(0, np.min(control_effect[:, :, i_train]))

        plt.figure(figsize=(3.2, 2.5), dpi=300)
        cmap = plt.get_cmap('plasma')
        colors = [cmap(i) for i in np.linspace(0, 1, len(target_layers))]
        for i, layer in enumerate(target_layers):
            plt.plot(control_effect[i, :, i_train], label=f'{layer+1}', color=colors[i])
            plt.fill_between(affected_layers, lower[i, :, i_train], upper[i, :, i_train], color=colors[i], alpha=0.07)
            plt.scatter(layer, control_effect[i, layer, i_train], color=colors[i], marker='o', s=40, linewidths=1, edgecolors='white')

        plt.xlabel('Layer of projected activation')
        plt.xticks([0, max_layer // 2, max_layer], ['1', str(max_layer // 2 + 1), str(max_layer + 1)])
        plt.ylabel(f'Control Effect ({effect_unit})')
        plt.ylim(bottom=y_min)
        plt.legend(fontsize=6, title='Target layer', title_fontsize=6, loc='upper left', bbox_to_anchor=(0, 1.05))
        plt.tight_layout()
        plt.savefig(f"{save_file}_control_effect_progress.{fig_format}", **PLOT_PARAMS)
        plt.close()
