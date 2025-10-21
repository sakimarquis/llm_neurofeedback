import os
from joblib import load
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from scipy.spatial.distance import cosine
from scipy.special import softmax
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import re

from analysis.stats_fn import compute_snr, compute_mean_sem, calc_cohen_d
from utils import load_exp_cfg, set_mpl, PLOT_PARAMS, safe_dump, load_saved_data
from configs.settings import SELECTED_LAYERS


def get_scores_and_hiddens(all_scores, layer):
    """get the average scores and hiddens for the given layer"""
    filter_cond0_label0 = all_scores[(all_scores['imitate_label'] == 0) & (all_scores['flip_shown_label'] == 0)]
    filter_cond1_label1 = all_scores[(all_scores['imitate_label'] == 1) & (all_scores['flip_shown_label'] == 0)]
    filter_cond1_label0 = all_scores[(all_scores['imitate_label'] == 0) & (all_scores['flip_shown_label'] == 1)]
    filter_cond0_label1 = all_scores[(all_scores['imitate_label'] == 1) & (all_scores['flip_shown_label'] == 1)]

    # [n_exp, n_dim]
    neural_cond0_label0 = np.array([x[layer] for x in filter_cond0_label0['processed_hiddens'].to_list()]).squeeze()
    neural_cond1_label1 = np.array([x[layer] for x in filter_cond1_label1['processed_hiddens'].to_list()]).squeeze()
    neural_cond1_label0 = np.array([x[layer] for x in filter_cond1_label0['processed_hiddens'].to_list()]).squeeze()
    neural_cond0_label1 = np.array([x[layer] for x in filter_cond0_label1['processed_hiddens'].to_list()]).squeeze()

    imit_cond0_label0 = filter_cond0_label0['imitate_example_scores'].to_numpy()
    imit_cond1_label1 = filter_cond1_label1['imitate_example_scores'].to_numpy()
    imit_cond1_label0 = filter_cond1_label0['imitate_example_scores'].to_numpy()
    imit_cond0_label1 = filter_cond0_label1['imitate_example_scores'].to_numpy()
    imit_0_scores = (imit_cond0_label0 + imit_cond0_label1) / 2
    imit_1_scores = (imit_cond1_label1 + imit_cond1_label0) / 2

    return neural_cond0_label0, neural_cond1_label1, neural_cond1_label0, neural_cond0_label1, imit_0_scores, imit_1_scores


def compute_scores_and_hiddens_diff_on_other_pc(classifier, neural_imit_0_noflip, neural_imit_1_noflip,
                                                neural_imit_0_flip, neural_imit_1_flip, n_other_pc=512):
    neural_diff = (neural_imit_1_noflip + neural_imit_0_flip) / 2 - (neural_imit_0_noflip + neural_imit_1_flip) / 2
    score_0 = np.zeros((neural_diff.shape[0], n_other_pc), dtype=np.float32)
    score_1 = np.zeros_like(score_0)
    all_neural_diff = np.zeros_like(score_0)
    for i_pc_to_compare in range(n_other_pc):
        classifier.pc_number = i_pc_to_compare + 1  # set the pc number of the classifier
        # compute the scores on the selected pc
        imit_1_score = classifier.decision_function(neural_imit_1_noflip) + classifier.decision_function(neural_imit_0_flip)
        imit_0_score = classifier.decision_function(neural_imit_0_noflip) + classifier.decision_function(neural_imit_1_flip)
        # cosine similarity between neural_diff and selected pc
        hidden_cos = np.array([1 - cosine(neural_diff[i], classifier.axis_) for i in range(len(neural_diff))])
        score_0[:, i_pc_to_compare] = imit_0_score
        score_1[:, i_pc_to_compare] = imit_1_score
        all_neural_diff[:, i_pc_to_compare] = hidden_cos
    return score_0, score_1, all_neural_diff


def compute_scores_and_hiddens_diff_lr(classifier, neural_imit_0_noflip, neural_imit_1_noflip,
                                       neural_imit_0_flip, neural_imit_1_flip):
    neural_diff = (neural_imit_1_noflip + neural_imit_0_flip) / 2 - (neural_imit_0_noflip + neural_imit_1_flip) / 2
    imit_1_score = classifier.decision_function(neural_imit_1_noflip) + classifier.decision_function(neural_imit_0_flip)
    imit_0_score = classifier.decision_function(neural_imit_0_noflip) + classifier.decision_function(neural_imit_1_flip)
    hidden_cos = np.array([1 - cosine(neural_diff[i], classifier.coef_[0]) for i in range(len(neural_diff))])
    return imit_0_score, imit_1_score, hidden_cos


def plot_target_on_affected(mean, se, n_train_examples, target_pcs, affected_pcs, colors, y_label, save_file):
    font_size = 7
    is_lr = True if target_pcs == [-1] else False  # hard code the lr plot
    for i, pc_number in enumerate(target_pcs):
        plt.figure(figsize=(2.5, 1.75), dpi=300)

        if is_lr:
            plt.title(f'Target axis: LR', fontsize=font_size)
            plt.plot(n_train_examples, mean[0, : , -1], alpha=1, color='red', linewidth=1.5, label=f'LR')
            lower_bound = mean[0,:, -1] - se[0,:, -1]
            upper_bound = mean[0,:, -1] + se[0,:, -1]
            plt.fill_between(n_train_examples, lower_bound, upper_bound, alpha=0.2, color='red')
        else:
            plt.title(f'Target axis: PC{pc_number}', fontsize=font_size)

        for j, pc_to_compare in enumerate(affected_pcs):
            color = colors[j]
            line_style = 'solid' if pc_to_compare == pc_number else 'dotted'
            plt.plot(n_train_examples, mean[i, : ,pc_to_compare - 1], alpha=1, color=color, linewidth=1.5,
                    linestyle=line_style, label=f'PC{pc_to_compare}')
            lower_bound = mean[i,:,pc_to_compare - 1] - se[i,:,pc_to_compare - 1]
            upper_bound = mean[i,:,pc_to_compare - 1] + se[i,:,pc_to_compare - 1]
            plt.fill_between(n_train_examples, lower_bound, upper_bound, alpha=0.2, color=color)
        plt.xlabel('# Examples', fontsize=font_size)
        plt.ylabel(y_label, fontsize=font_size)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.legend(bbox_to_anchor=(0.95, 1.1), fontsize=font_size-1.5, title='Affected axis', title_fontsize=font_size-0.5)
        plt.tight_layout()
        if is_lr:
            plt.savefig(f"{save_file}.{fig_format}", **PLOT_PARAMS)
        else:
            plt.savefig(f"{save_file}_pc{pc_number}.{fig_format}", **PLOT_PARAMS)
        plt.close()


def plot_main_effect(pc_effect, lr_effect, n_train_examples, target_pcs, colors, y_label, save_file):
    font_size = 7
    plt.figure(figsize=(2.5, 1.75), dpi=300)
    plt.plot(n_train_examples, lr_effect[0, :, -1], alpha=0.9, color='red', linewidth=1.5, label=f'LR')
    for i, pc_number in enumerate(target_pcs):
        plt.plot(n_train_examples, pc_effect[i, :, pc_number - 1], alpha=0.9, color=colors[i], linewidth=1.5,
                 linestyle='solid', label=f'PC{pc_number}')
    plt.xlabel('# Examples', fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.legend(bbox_to_anchor=(0.95, 0.9), fontsize=font_size - 1.5)
    plt.tight_layout()
    plt.savefig(f"{save_file}_main_effect.{fig_format}", **PLOT_PARAMS)
    plt.close()


def plot_layers_pc_control_precision(fig_dir, method='cohen_d'):
    """Plot the target effect of the imitation score difference and neural difference"""
    cmap = plt.get_cmap('plasma')
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 2), dpi=300, sharey=True)

    all_legend_lines = []
    all_legend_labels = []

    for mode_idx, mode in enumerate(['active', 'inactive']):
        save_file = f"{fig_dir}/pcascore_{mode}"
        if method == 'cohen_d':
            all_score_1 = load(f"{save_file}_all_score_1.pkl")
            all_score_0 = load(f"{save_file}_all_score_0.pkl")
            all_effects, _ = calc_cohen_d(all_score_1, all_score_0, axis=2)
        elif method == 'neural_similarity':
            imit_neuro_diff = load(f"{save_file}_neuro_diff.pkl")
            all_effects = 1 - imit_neuro_diff
        else:
            raise ValueError("method should be 'cohen_d' or 'neural_similarity'")

        all_control_precision = []

        for i, layer in enumerate(layers):
            control_precision = []
            for i_pc, pc in enumerate(pc_to_control):
                effect = all_effects[i_pc, i, -1]
                precision = np.abs(effect[pc - 1]) / np.abs(effect).mean(0)
                control_precision.append(precision)
            all_control_precision.append(control_precision)

        ax = axes[mode_idx]
        for i, layer in enumerate(layers):
            line, = ax.plot(range(len(pc_to_control)), all_control_precision[i],
                            label=f'Layer {layer + 1}', alpha=0.8,
                            color=cmap(i / (len(layers) - 1)))
            if mode_idx == 0:
                all_legend_lines.append(line)
                all_legend_labels.append(f'Layer {layer + 1}')
        mean_line, = ax.plot(range(len(pc_to_control)),
                             np.mean(all_control_precision, axis=0), 'k--',
                             label='Layer mean', alpha=0.8)
        if mode_idx == 0:
            all_legend_lines.append(mean_line)
            all_legend_labels.append('Layer mean')

        ax.set_xticks(range(len(pc_to_control)))
        ax.set_xticklabels([f'{pc}' for pc in pc_to_control])
        ax.set_xlabel('Target PC axis')
        ax.set_ylabel('Control precision')
        ax.set_ylim(0.1, 10)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_yscale('log')
        ax.set_title('Explicit control' if mode == 'active' else 'Implicit control', fontsize=8)

    fig.legend(handles=all_legend_lines, labels=all_legend_labels, loc='center left',
               fontsize=6.5, bbox_to_anchor=(0.8, 0.6), bbox_transform=fig.transFigure)
    plt.tight_layout(rect=(0, 0, 0.83, 1))  # leave space on the right for the legend
    plt.savefig(f"{fig_dir}/control_precision_{method}.{fig_format}", **PLOT_PARAMS)
    plt.close()



def plot_snr_heatmap(snr, layer, pc_to_control, save_file):
    font_size = 10
    fig, ax = plt.subplots(figsize=(4.25, 3), dpi=300)
    mask = np.isnan(snr)
    abs_max = np.abs(snr[1:, 1:]).max()
    if snr[0, 0] > abs_max:
        abs_max = (snr[0, 0] + abs_max) / 2  # not too large
    sns.heatmap(snr, mask=mask, ax=ax, annot=True, fmt=".2f", cmap='bwr', center=0, vmin=-abs_max, vmax=abs_max,
                cbar_kws={"shrink": .7}, annot_kws={"color": "black", "fontsize": font_size-2})
    for i in range(snr.shape[1]):
        rect = patches.Rectangle((i, i), 1, 1, linewidth=1.2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
    ax.invert_yaxis()
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)
    if snr.shape[0] == len(pc_to_control):
        ax.set_xticks(np.arange(snr.shape[0]) + 0.5, [f'PC{pc}' for pc in pc_to_control])
        ax.set_yticks(np.arange(snr.shape[0]) + 0.5, [f'PC{pc}' for pc in pc_to_control], rotation=0)
    else:
        ax.set_xticks(np.arange(snr.shape[0]) + 0.5, ['LR'] + [f'PC{pc}' for pc in pc_to_control])
        ax.set_yticks(np.arange(snr.shape[0]) + 0.5, ['LR'] + [f'PC{pc}' for pc in pc_to_control], rotation=0)
    ax.set_xlabel('Affected axis', fontsize=font_size)
    ax.set_ylabel('Target axis', fontsize=font_size)
    if isinstance(layer, int):
        ax.set_title(f'Control effect ({effect_unit}): layer {layer+1}', fontsize=font_size)
    else:
        ax.set_title(f'Control effect ({effect_unit}): layer {layer}', fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size - 2)
    plt.tight_layout()
    plt.savefig(f"{save_file}_layer{layer}_snr_heatmap.{fig_format}", **PLOT_PARAMS)
    plt.close()


def plot_layers_relative_snr(layers, score_diff_snr, pc_to_control, save_file):
    plt.figure(figsize=(2, 1.5), dpi=300)
    cmap = plt.get_cmap('plasma')
    for i_layer, layer in enumerate(layers):
        snr = score_diff_snr[:, i_layer, -1][:, np.array(pc_to_control) - 1]
        p = softmax(np.abs(snr))
        plt.plot(np.diag(p), label=f'Layer {layer+1}', color=cmap(i_layer / (len(layers) - 1)), alpha=0.8)
    plt.legend(fontsize=5)
    plt.ylabel(f'Relative control effect ({effect_unit})', fontsize=6)
    plt.xlabel('PC axis', fontsize=6)
    plt.xticks(np.arange(len(pc_to_control)), [f'{pc}' for pc in pc_to_control], fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{save_file}_relative_snr.{fig_format}", **PLOT_PARAMS)
    plt.close()


def calculate_score_lr():
    pc_classifiers = load(save_dir / f"hidden_{cfg.process_hidden_method}_classifiers_pcascore_pc1.pkl")
    lr_classifiers = load(save_dir / f"hidden_{cfg.process_hidden_method}_classifiers_lr.pkl")

    for mode in ['active', 'inactive']:
        save_file = f"{fig_dir}/lr_{mode}"
        imit_scorediff = np.zeros((len(layers), n_exp, len(n_train_examples), len(pc_to_compare) + 1))  # pc and lr
        imit_neurodiff_cos = np.zeros_like(imit_scorediff)
        imit_score = np.zeros((len(layers), n_exp, len(n_train_examples), 2))
        all_score_0 = np.zeros_like(imit_scorediff)
        all_score_1 = np.zeros_like(imit_scorediff)

        for i_layer, layer in enumerate(layers):
            pc_classifier = pc_classifiers[layer]
            lr_classifier = lr_classifiers[layer]

            for i_train_example, n_train in enumerate(n_train_examples):
                file_name = f'clf_lr_layer{layer}_ntrain{n_train}_mode{mode}'
                try:
                    all_scores = load_saved_data(file_name, save_dir, start=0, end=0 + n_exp)
                except ValueError:  # no objects to concatenate
                    print(f"File {file_name} not found.")
                    continue

                (neural_imit_0_noflip, neural_imit_1_noflip, neural_imit_0_flip, neural_imit_1_flip,
                    imit_0_scores, imit_1_scores) = get_scores_and_hiddens(all_scores, layer)
                imit_score[i_layer, :, i_train_example, 0] = imit_0_scores
                imit_score[i_layer, :, i_train_example, 1] = imit_1_scores
                score_0, score_1, all_neural_diff = compute_scores_and_hiddens_diff_on_other_pc(
                        pc_classifier, neural_imit_0_noflip, neural_imit_1_noflip, neural_imit_0_flip, neural_imit_1_flip)
                all_score_diff = (score_1 - score_0) / 2
                imit_scorediff[i_layer, :all_score_diff.shape[0], i_train_example, :-1] = all_score_diff
                imit_neurodiff_cos[i_layer, :all_neural_diff.shape[0], i_train_example, :-1] = all_neural_diff
                all_score_0[i_layer, :score_0.shape[0], i_train_example, :-1] = score_0
                all_score_1[i_layer, :score_1.shape[0], i_train_example, :-1] = score_1

                lr_score_0, lr_score_1, lr_neural_diff = compute_scores_and_hiddens_diff_lr(
                        lr_classifier, neural_imit_0_noflip, neural_imit_1_noflip, neural_imit_0_flip, neural_imit_1_flip)
                lr_score_diff = (lr_score_1 - lr_score_0) / 2
                imit_scorediff[i_layer, :lr_score_diff.shape[0], i_train_example, -1] = lr_score_diff
                imit_neurodiff_cos[i_layer, :lr_neural_diff.shape[0], i_train_example, -1] = lr_neural_diff
                all_score_0[i_layer, :lr_score_0.shape[0], i_train_example, -1] = lr_score_0
                all_score_1[i_layer, :lr_score_1.shape[0], i_train_example, -1] = lr_score_1

        safe_dump(imit_scorediff, f"{save_file}_score_diff.pkl")
        safe_dump(imit_neurodiff_cos, f"{save_file}_neuro_diff.pkl")
        safe_dump(imit_score, f"{save_file}_score.pkl")
        safe_dump(all_score_0, f"{save_file}_all_score_0.pkl")
        safe_dump(all_score_1, f"{save_file}_all_score_1.pkl")


def calculate_score_pca():
    pc_classifiers = load(save_dir / f"hidden_{cfg.process_hidden_method}_classifiers_pcascore_pc1.pkl")
    for mode in ['active', 'inactive']:
        save_file = f"{fig_dir}/{cfg.clf}_{mode}"
        imit_scorediff = np.zeros((len(pc_to_control), len(layers), n_exp, len(n_train_examples), len(pc_to_compare)))
        imit_neurodiff_cos = np.zeros_like(imit_scorediff)
        imit_score = np.zeros((len(pc_to_control), len(layers), n_exp, len(n_train_examples), 2))
        all_score_0 = np.zeros_like(imit_scorediff)
        all_score_1 = np.zeros_like(imit_scorediff)

        for i_layer, layer in enumerate(layers):
            for i_pc, pc_number in enumerate(pc_to_control):
                classifier = pc_classifiers[layer]
                classifier.pc_number = pc_number

                for i_train_example, n_train in enumerate(n_train_examples):
                    file_name = f'clf_{cfg.clf}_pc{pc_number}_layer{layer}_ntrain{n_train}_mode{mode}'
                    try:
                        all_scores = load_saved_data(file_name, save_dir, start=0, end=0 + n_exp)
                    except ValueError:  # no objects to concatenate
                        print(f"File {file_name} not found.")
                        continue

                    (neural_imit_0_noflip, neural_imit_1_noflip, neural_imit_0_flip, neural_imit_1_flip,
                        imit_0_scores, imit_1_scores) = get_scores_and_hiddens(all_scores, layer)
                    imit_score[i_pc, i_layer, :imit_0_scores.shape[0], i_train_example, 0] = imit_0_scores
                    imit_score[i_pc, i_layer, :imit_1_scores.shape[0], i_train_example, 1] = imit_1_scores
                    score_0, score_1, all_neural_diff = compute_scores_and_hiddens_diff_on_other_pc(
                            classifier, neural_imit_0_noflip, neural_imit_1_noflip, neural_imit_0_flip, neural_imit_1_flip)
                    all_score_diff = (score_1 - score_0) / 2
                    if not np.allclose(imit_1_scores - imit_0_scores, all_score_diff[:, pc_number - 1], rtol=0.01):
                        print(f"===Warning: do not match for {cfg.model_name} layer{layer} pc{pc_number} n_train{n_train} mode{mode}===")
                    all_score_0[i_pc, i_layer, :score_0.shape[0], i_train_example, :] = score_0
                    all_score_1[i_pc, i_layer, :score_1.shape[0], i_train_example, :] = score_1
                    imit_scorediff[i_pc, i_layer, :all_score_diff.shape[0], i_train_example, :] = all_score_diff
                    imit_neurodiff_cos[i_pc, i_layer, :all_neural_diff.shape[0], i_train_example, :] = all_neural_diff

        print((imit_scorediff==0).sum(), 'of imit_scorediff are all zeros')
        safe_dump(imit_scorediff, f"{save_file}_score_diff.pkl")
        safe_dump(imit_neurodiff_cos, f"{save_file}_neuro_diff.pkl")
        safe_dump(imit_score, f"{save_file}_score.pkl")
        safe_dump(all_score_0, f"{save_file}_all_score_0.pkl")
        safe_dump(all_score_1, f"{save_file}_all_score_1.pkl")


def plot_imit_hist(layers, pc_to_control, n_train_examples, imit_scores, snr, save_file):
    n_bins = 25

    for i_layer, layer in enumerate(layers):
        for i_pc, pc_number in enumerate(pc_to_control):
            fig, axes = plt.subplots(3, n_cols, figsize=(8, 6), dpi=300)

            for i_train_example, n_train in enumerate(n_train_examples):
                imit_0_scores = imit_scores[i_pc, i_layer, :, i_train_example, 0]
                imit_1_scores = imit_scores[i_pc, i_layer, :, i_train_example, 1]

                # cohen_d, ci = calc_cohen_d(imit_1_scores, imit_0_scores)
                t_stat, p_value = ttest_ind(imit_1_scores, imit_0_scores)
                ax = axes[i_train_example // n_cols, i_train_example % n_cols]
                ax.hist(imit_0_scores, label='Imitate <0>', alpha=0.6, bins=n_bins)
                ax.hist(imit_1_scores, label='Imitate <1>', alpha=0.6, bins=n_bins)
                if pc_number == -1:
                    cond = 'LR'
                    snr_value = snr[i_pc, i_layer, i_train_example, -1]
                else:
                    cond = f'PC{pc_number}'
                    snr_value = snr[i_pc, i_layer, i_train_example, pc_number - 1]
                if p_value <= 0.01:
                    exponent = int(np.floor(np.log10(p_value)))
                    ax.set_title(rf'${cond}: N={n_train}, {effect_unit}={snr_value:.2f}, p<=10^{{{exponent}}}$')
                else:
                    ax.set_title(rf'${cond}: N={n_train}, {effect_unit}={snr_value:.2f}, p={p_value:.2f}$')
                ax.legend(handletextpad=0.4)
                ax.set_xlabel('Scores')
                ax.set_ylabel('Frequency')
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

            plt.tight_layout()
            if pc_number == -1:
                plt.savefig(f"{save_file}_hist_layer{layer}.{fig_format}", **PLOT_PARAMS)
            else:
                plt.savefig(f"{save_file}_hist_layer{layer}_pc{pc_number}.{fig_format}", **PLOT_PARAMS)
            plt.close()


def plot_models_control_effect_lr():
    dataset_name, label_name = "commonsense", "labels"
    layers_snr = {}
    layers_snr_upper = {}
    layers_snr_lower = {}
    model_size = 4
    red_cmap = plt.get_cmap('Reds_r')
    reds = [red_cmap(i / model_size) for i in range(model_size)]
    blue_cmap = plt.get_cmap('Blues_r')
    blues = [blue_cmap(i / model_size) for i in range(model_size)]
    colors = {'llama3.1_70b': reds[0], 'llama3.1_8b': reds[1], 'llama3.2_3b': reds[2], 'llama3.2_1b': reds[3],
              'qwen2.5_7b': blues[1], 'qwen2.5_3b': blues[2], 'qwen2.5_1.5b': blues[3]}

    for model in ["llama3.1_70b", "llama3.1_8b", "llama3.2_3b", "llama3.2_1b", "qwen2.5_7b", "qwen2.5_3b", "qwen2.5_1.5b"]:
        layers_snr[model] = {}
        layers_snr_upper[model] = {}
        layers_snr_lower[model] = {}
        cfg = load_exp_cfg(model)
        save_dir = Path("results") / (cfg.model_name.replace("/", "_")) / dataset_name
        fig_dir = f'{save_dir}/imitation'
        layers = SELECTED_LAYERS[model]

        for mode in ['active', 'inactive']:
            layers_snr[model][mode] = []
            layers_snr_upper[model][mode] = []
            layers_snr_lower[model][mode] = []

            if use_paired_samples:
                lr_imit_score_diff = load(f"{fig_dir}/lr_{mode}_score_diff.pkl")[None, ...]
                lr_score_diff_snr, lr_score_diff_snr_se = compute_snr(lr_imit_score_diff, axis=2)  # average over experiments
            else:
                lr_score_0 = load(f"{fig_dir}/lr_{mode}_all_score_0.pkl")[None, ...]
                lr_score_1 = load(f"{fig_dir}/lr_{mode}_all_score_1.pkl")[None, ...]
                lr_score_diff_snr, lr_score_diff_snr_se = calc_cohen_d(lr_score_1, lr_score_0, axis=2)

            for i_layer, layer in enumerate(layers):
                layers_snr[model][mode].append(lr_score_diff_snr[0, i_layer, -1, -1])
                layers_snr_upper[model][mode].append(lr_score_diff_snr[0, i_layer, -1, -1] + lr_score_diff_snr_se[0, i_layer, -1, -1])
                layers_snr_lower[model][mode].append(lr_score_diff_snr[0, i_layer, -1, -1] - lr_score_diff_snr_se[0, i_layer, -1, -1])

    fig, ax = plt.subplots(1, 2, figsize=(5, 2), dpi=300)  # , sharey=True
    for j, mode in enumerate(layers_snr[model].keys()):
        for i, model in enumerate(layers_snr.keys()):
            ax[j].plot(layers_snr[model][mode], label=model, alpha=0.7, color=colors[model])
            ax[j].fill_between(range(len(layers)), layers_snr_lower[model][mode], layers_snr_upper[model][mode],
                               alpha=0.2, color=colors[model])
        ax[j].set_xlabel('Layer (quantile)')
        ax[j].set_ylabel(f'Control effect ({effect_unit})')
        ax[j].set_xticks(range(5), [f'{i/4}' for i in range(5)])
    ax[0].set_title('LR: explicit control')
    ax[1].set_title('LR: implicit control')
    legend_elements = [
        Line2D([0], [0], color=colors[model], lw=1.5, label=model)
        for model in layers_snr.keys()
    ]
    plt.tight_layout(rect=(0, 0, 0.83, 1))  # right=0.85 for legend
    fig.legend(handles=legend_elements, loc='center left', fontsize=6.5, bbox_transform=fig.transFigure,
               bbox_to_anchor=(0.8, 0.55))  # Adjust the x-coordinate to move the legend to the right
    plt.savefig(f"results/models_layers_lr_snr.{fig_format}", **PLOT_PARAMS)
    plt.close()
    plot_control_effect_vs_model_size(layers_snr)


def plot_layers_snr(score_diff_snr, score_diff_snr_se, lr_score_diff_snr, lr_score_diff_snr_se, layers, pc_to_control, save_file):
    early_pc_idx = np.array([0, 1, 2, 3], dtype=int)
    late_pc_idx = np.array([4, 5, 6], dtype=int)
    all_pcs = np.array(pc_to_control)
    early_pc_snr = np.zeros(len(layers))
    early_pc_snr_se = np.zeros(len(layers))
    late_pc_snr = np.zeros(len(layers))
    late_pc_snr_se = np.zeros(len(layers))
    lr_snr = np.zeros(len(layers))
    lr_snr_se = np.zeros(len(layers))
    for i_layer, layer in enumerate(layers):
        early_pc_snr[i_layer] = score_diff_snr[early_pc_idx, i_layer, -1, all_pcs[early_pc_idx] - 1].mean()
        early_pc_snr_se[i_layer] = score_diff_snr_se[early_pc_idx, i_layer, -1, all_pcs[early_pc_idx] - 1].mean()
        late_pc_snr[i_layer] = score_diff_snr[late_pc_idx, i_layer, -1, all_pcs[late_pc_idx] - 1].mean()
        late_pc_snr_se[i_layer] = score_diff_snr_se[late_pc_idx, i_layer, -1, all_pcs[late_pc_idx] - 1].mean()
        lr_snr[i_layer] = lr_score_diff_snr[0, i_layer, -1, -1]
        lr_snr_se[i_layer] = lr_score_diff_snr_se[0, i_layer, -1, -1]

    plt.figure(figsize=(2, 1.75), dpi=300)
    font_size = 7
    plt.plot(lr_snr, label='LR', color='red', alpha=0.8)
    plt.fill_between(range(len(layers)), lr_snr - lr_snr_se, lr_snr + lr_snr_se, alpha=0.2, color='red')
    plt.plot(early_pc_snr, label='Early PCs', color='blue', alpha=0.8)
    plt.fill_between(range(len(layers)), early_pc_snr - early_pc_snr_se, early_pc_snr + early_pc_snr_se, alpha=0.2, color='blue')
    plt.plot(late_pc_snr, label='Late PCs', color='green', alpha=0.8)
    plt.fill_between(range(len(layers)), late_pc_snr - late_pc_snr_se, late_pc_snr + late_pc_snr_se, alpha=0.2, color='green')
    plt.xticks(range(len(layers)), [f'{i+1}' for i in layers], fontsize=font_size)
    plt.xlabel('Layer', fontsize=font_size)
    plt.ylabel(f'Control effect ({effect_unit})', fontsize=font_size)
    plt.legend(fontsize=5, loc='upper left', bbox_to_anchor=(0, 1.1))
    plt.tight_layout()
    plt.savefig(f"{save_file}_layers_snr.{fig_format}", **PLOT_PARAMS)
    plt.close()


def extract_model_size(model_name: str) -> float:
    """Extract numeric model size from a model name like 'llama3.1_70b'."""
    match = re.search(r'_(\d+(?:\.\d+)?)b', model_name)
    return float(match.group(1)) if match else None


def linear_fit_with_ci(x, y, n_boot=1000, ci=95):
    """Fit linear model and estimate CI band using bootstrapping."""
    x_log = np.log10(x).reshape(-1, 1)
    y_log = np.log10(y)
    model = LinearRegression().fit(x_log, y_log)
    y_pred = model.predict(x_log)

    # Bootstrap confidence interval
    preds = []
    for _ in range(n_boot):
        x_samples, y_samples = resample(x_log, y_log)
        m = LinearRegression().fit(x_samples, y_samples)
        preds.append(m.predict(x_log))
    preds = np.array(preds)
    lower = np.percentile(preds, (100 - ci) / 2, axis=0)
    upper = np.percentile(preds, 100 - (100 - ci) / 2, axis=0)
    return y_pred, lower, upper


def plot_control_effect_vs_model_size(layers_snr):
    families = ['llama', 'qwen']
    modes = ['active', 'inactive']
    colors = {'llama': 'C0', 'qwen': 'C1'}

    fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=300, sharey=True)

    for i, mode in enumerate(modes):
        ax = axes[i]
        for fam in families:
            xs, ys = [], []
            for model_name, mode_dict in layers_snr.items():
                if fam not in model_name:
                    continue
                size = extract_model_size(model_name)
                if size is None or mode not in mode_dict:
                    continue
                control_vals = mode_dict[mode]
                avg_val = np.mean(control_vals)
                xs.append(size)
                ys.append(avg_val)

            xs = np.array(xs)
            ys = np.array(ys)
            y_pred, lower, upper = linear_fit_with_ci(xs, ys)
            sort_idx = np.argsort(xs)
            xs_sorted = xs[sort_idx]
            y_pred = y_pred[sort_idx]
            lower = lower[sort_idx]
            upper = upper[sort_idx]
            ax.scatter(xs, ys, label=f'{fam}', color=colors[fam], s=30)
            ax.plot(xs_sorted, 10**y_pred, color=colors[fam], label=f'{fam} fit')
            ax.fill_between(xs_sorted, 10**lower, 10**upper, color=colors[fam], alpha=0.2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Model Size (B params)')
        if i == 0:
            ax.set_ylabel(f'Avg. Control Effect ({effect_unit})')
        ax.set_title(f'{mode.capitalize()} control')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    set_mpl()
    model = "llama3.2_3b"  # "llama3.1_8b" or "qwen2.5_7b" or "llama3.1_70b" or "llama3.2_1b" or "qwen2.5_1.5b" or "llama3.2_3b" or "qwen2.5_3b"
    cfg = load_exp_cfg(model)
    dataset_name, label_name = "sycophancy", "labels"
    save_dir = Path("results") / (cfg.model_name.replace("/", "_")) / dataset_name
    fig_dir = f'{save_dir}/imitation'
    os.makedirs(fig_dir, exist_ok=True)
    n_train_examples = cfg.n_train_examples
    cmap = plt.get_cmap('viridis')
    n_cols = 3
    n_exp = 100

    layers = SELECTED_LAYERS[model]  # [15]
    pc_to_control = cfg.all_pc_exp
    pc_to_compare = [i for i in range(512)]
    pc_positions = np.linspace(1, 0, len(pc_to_control))  # we compare control PCs with control PCs
    colors = [cmap(p) for p in pc_positions]

    use_paired_samples = False
    effect_unit = 'SNR' if use_paired_samples else 'd'
    fig_format = 'svg'
    # calculate_score_pca()
    # calculate_score_lr()
    # plot_models_control_effect_lr()
    plot_layers_pc_control_precision(fig_dir, method='cohen_d')

    for mode in ['active', 'inactive']:
        save_file = f"{fig_dir}/{cfg.clf}_{mode}"
        imit_neurodiff_cos = load(f"{save_file}_neuro_diff.pkl")
        pca_imit_score = load(f"{save_file}_score.pkl")
        lr_imit_neuro_diff = load(f"{fig_dir}/lr_{mode}_neuro_diff.pkl")[None, ...]
        lr_imit_score = load(f"{fig_dir}/lr_{mode}_score.pkl")[None, ...]

        if use_paired_samples:
            imit_scorediff = load(f"{save_file}_score_diff.pkl")
            lr_imit_score_diff = load(f"{fig_dir}/lr_{mode}_score_diff.pkl")[None, ...]
            score_diff_snr, score_diff_snr_se = compute_snr(imit_scorediff, axis=2)  # average over experiments
            lr_score_diff_snr, lr_score_diff_snr_se = compute_snr(lr_imit_score_diff, axis=2)
            score_diff_snr_all, score_diff_snr_se_all = compute_snr(imit_scorediff, axis=(1, 2))  # average over layers and experiments
            lr_score_diff_snr_all, lr_score_diff_snr_se_all = compute_snr(lr_imit_score_diff, axis=(1, 2))
            print(f'SD of score_diff in {mode} mode: {np.std(imit_scorediff, axis=2).mean():.3f}')
        else:
            all_score_1 = load(f"{save_file}_all_score_1.pkl")  # (7, 5, 100, 9, 512), (pc, layer, exp, n_train, pc)
            all_score_0 = load(f"{save_file}_all_score_0.pkl")
            lr_score_1 = load(f"{fig_dir}/lr_{mode}_all_score_1.pkl")[None, ...]
            lr_score_0 = load(f"{fig_dir}/lr_{mode}_all_score_0.pkl")[None, ...]
            score_diff_snr, score_diff_snr_se = calc_cohen_d(all_score_1, all_score_0, axis=2)
            lr_score_diff_snr, lr_score_diff_snr_se = calc_cohen_d(lr_score_1, lr_score_0, axis=2)
            score_diff_snr_all, score_diff_snr_se_all = calc_cohen_d(all_score_1, all_score_0, axis=(1, 2))
            lr_score_diff_snr_all, lr_score_diff_snr_se_all = calc_cohen_d(lr_score_1, lr_score_0, axis=(1, 2))

        plot_layers_snr(score_diff_snr, score_diff_snr_se, lr_score_diff_snr, lr_score_diff_snr_se, layers, pc_to_control, save_file)
        plot_imit_hist(layers, pc_to_control, n_train_examples, pca_imit_score, score_diff_snr, save_file)
        plot_imit_hist(layers, [-1], n_train_examples, lr_imit_score, lr_score_diff_snr, save_file.replace('pcascore', 'lr'))

        y_label = f'Control effect ({effect_unit})'
        for i_layer, layer in enumerate(layers):
            output_fname = f"{save_file}_layer{layer}_score_diff"
            plot_target_on_affected(score_diff_snr[:, i_layer], score_diff_snr_se[:, i_layer], n_train_examples, pc_to_control, pc_to_control, colors, y_label, output_fname)
            plot_target_on_affected(lr_score_diff_snr[:, i_layer], lr_score_diff_snr_se[:, i_layer], n_train_examples, [-1], pc_to_control, colors, y_label, output_fname.replace('pcascore', 'lr'))
            plot_main_effect(score_diff_snr[:, i_layer], lr_score_diff_snr[:, i_layer], n_train_examples, pc_to_control, colors, y_label, f"{save_file}_layer{layer}")
        plot_target_on_affected(score_diff_snr_all, score_diff_snr_se_all, n_train_examples, pc_to_control, pc_to_control, colors, y_label, f"{save_file}_score_diff")
        plot_target_on_affected(lr_score_diff_snr_all, lr_score_diff_snr_se_all, n_train_examples, [-1], pc_to_control, colors, y_label, f"{save_file.replace('pcascore', 'lr')}_score_diff")
        plot_main_effect(score_diff_snr_all, lr_score_diff_snr_all, n_train_examples, pc_to_control, colors, y_label, f"{save_file}")


        neuro_diff_mean, neuro_diff_sem = compute_mean_sem(imit_neurodiff_cos, axis=2)
        y_label = r'cos($\Delta\mathrm{hiddens}$, PC axis)'
        for i_layer, layer in enumerate(layers):
            output_fname = f"{save_file}_layer{layer}_neuro_diff"
            plot_target_on_affected(neuro_diff_mean[:, i_layer], neuro_diff_sem[:, i_layer], n_train_examples, pc_to_control, pc_to_control, colors, y_label, output_fname)
        neuro_diff_mean_all, neuro_diff_sem_all = compute_mean_sem(imit_neurodiff_cos, axis=(1, 2))
        plot_target_on_affected(neuro_diff_mean_all, neuro_diff_sem_all, n_train_examples, pc_to_control, pc_to_control, colors, y_label, f"{save_file}_neuro_diff")

        # only use the n_train_examples[-1] to plot the heatmap
        heatmap_snr = np.full((8, 8), np.nan)
        selected_directions = np.array(pc_to_control) - 1
        for i_layer, layer in enumerate(layers):
            snr = score_diff_snr[:, i_layer, -1, selected_directions]  # (7, 512), each pc on other pcs
            snr_lr = lr_score_diff_snr[:, i_layer, -1, selected_directions]  # (1, 513), lr on (pcs + lr)
            heatmap_snr[:, 1:] = np.concatenate((snr_lr, snr), axis=0)
            heatmap_snr[0, 0] = lr_score_diff_snr[0, i_layer, -1, -1]
            plot_snr_heatmap(heatmap_snr, layer, pc_to_control, save_file)
        snr = score_diff_snr_all[:, -1, selected_directions]
        snr_lr = lr_score_diff_snr_all[:, -1, selected_directions]
        heatmap_snr[:, 1:] = np.concatenate((snr_lr, snr), axis=0)
        heatmap_snr[0, 0] = lr_score_diff_snr_all[0, -1, -1]
        plot_snr_heatmap(heatmap_snr, 'mean', pc_to_control, save_file)
        plot_layers_relative_snr(layers, score_diff_snr, pc_to_control, save_file)
