import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import expit
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from configs.settings import SELECTED_LAYERS
from utils import load_exp_cfg, set_mpl, PLOT_PARAMS, load_saved_data


def binary_ce_loss(true_scores, prob_scores, eps=1e-8):
    """Compute the binary cross entropy loss for a set of predictions.
    :param true_scores: array of ground truth values (0 or 1)
    :param prob_scores: array of probabilities (between 0 and 1)
    :param eps: a small number to avoid log(0)
    """
    true_scores = np.array(true_scores)
    prob_scores = np.array(prob_scores)
    # Apply the binary cross entropy loss formula elementwise
    ce = - (true_scores * np.log(prob_scores + eps) + (1 - true_scores) * np.log(1 - prob_scores + eps))
    return ce


def get_loss_and_acc(model_s, n_layers, pcs, save_dir, n_exp=100, n_sample=600):
    acc = np.zeros((len(pcs), n_layers, n_exp, n_sample))  # add one for lr
    ce_loss = np.zeros((len(pcs), n_layers, n_exp, n_sample))  # add one for lr

    for i_pc, pc_number in enumerate(pcs):
        cfg_s = load_exp_cfg(model_s, pc_number=pc_number)
        if pc_number == -1:  # for lr
            file_name = 'lr'
        else:
            file_name = f'{cfg_s.clf}_{cfg_s.clf_name}'
        all_scores = load_saved_data(f"clf_{file_name}", save_dir, experiment='predict')
        all_scores['est_correct'] = all_scores.apply(lambda row: np.array(row['all_example_est_scores']) == np.array(row['all_example_true_scores']), axis=1)
        # from logit to pr
        all_scores['prob'] = all_scores['all_example_est_scores_logitdiff'].apply(lambda x: expit(np.array(x)))
        # from pr to loss
        all_scores['ce_loss'] = all_scores.apply(lambda row: binary_ce_loss(row['all_example_true_scores'], row['prob']), axis=1)

        layers = sorted(all_scores['layer'].unique())
        experiments = sorted(all_scores['experiment'].unique())

        for _, row in all_scores.iterrows():  # different layers
            i_layer = layers.index(row['layer'])
            i_exp = experiments.index(row['experiment'])
            acc[i_pc, i_layer, i_exp, :] = row['est_correct']
            ce_loss[i_pc, i_layer, i_exp, :] = row['ce_loss']

    print((ce_loss == 0).sum(), 'of data are all zeros')
    return acc, ce_loss


def plot_prediction_performance(acc, ce_loss, pcs, save_dir, colors, n_sample=600):
    acc_mean = np.mean(acc, axis=(1, 2))  # mean over layers and experiments
    acc_mean = gaussian_filter1d(acc_mean, 1, axis=1)
    acc_sem = np.std(acc, axis=(1, 2), ddof=1) / np.sqrt(acc.shape[1] * acc.shape[2])
    acc_sem = gaussian_filter1d(acc_sem, 1, axis=1)
    ce_loss_mean = np.mean(ce_loss, axis=(1, 2))
    ce_loss_mean = gaussian_filter1d(ce_loss_mean, 1, axis=1)
    ce_loss_std = np.std(ce_loss, axis=(1, 2), ddof=1) / np.sqrt(ce_loss.shape[1] * ce_loss.shape[2])
    ce_loss_std = gaussian_filter1d(ce_loss_std, 1, axis=1)


    fig, axes = plt.subplots(1, 2, figsize=(5, 2), dpi=300)
    for i, pc_number in enumerate(pcs):
        color = colors[i]
        axes[0].plot(np.arange(1, 1 + n_sample), acc_mean[i], alpha=0.9, color=color, linewidth=0.5)
        axes[1].plot(np.arange(1, 1 + n_sample), ce_loss_mean[i], alpha=0.9, color=color, linewidth=0.5)
        axes[0].fill_between(np.arange(1, 1 + n_sample), acc_mean[i] - acc_sem[i], acc_mean[i] + acc_sem[i], alpha=0.2, color=color)
        axes[1].fill_between(np.arange(1, 1 + n_sample), ce_loss_mean[i] - ce_loss_std[i], ce_loss_mean[i] + ce_loss_std[i], alpha=0.2, color=color)

    axes[0].set_ylabel('Accuracy')
    axes[1].set_ylabel('Cross-entropy')
    axes[0].set_ylim(0.5, 0.87)
    axes[1].set_ylim(0.35, 1.2)
    for j in range(2):
        axes[j].set_xlabel('# Examples')

    legend_elements = []
    for i, pc_number in enumerate(pcs):
        if pc_number == -1:
            legend_elements.append(Line2D([0], [0], color=colors[i], lw=1.5, label='LR'))
        else:
            legend_elements.append(Line2D([0], [0], color=colors[i], lw=1.5, label=f'PC{pcs[i]}'))

    plt.tight_layout(rect=(0, 0, 0.85, 1))
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.82, 0.6), bbox_transform=fig.transFigure)
    plt.savefig(f'{save_dir}/perf.{fig_format}', **PLOT_PARAMS)
    plt.close()


if __name__ == "__main__":
    set_mpl()
    fig_format = 'svg'
    dataset_name, label_name = "commonsense", "labels"
    model_s = "llama3.1_8b"  # model generate score: "llama3.1_8b" or "qwen2.5_7b" or "llama3.1_70b" or "qwen2.5_72b"
    model_e = "llama3.1_8b"  # model run prediction exp: "llama3.1_8b" or "qwen2.5_7b" or "llama3.1_70b" or "qwen2.5_72b"

    cfg_s = load_exp_cfg(model_s)
    cfg_e = load_exp_cfg(model_e)
    save_dir = Path("results") / (cfg_s.model_name.replace("/", "_") + '-'+ cfg_e.model_name.replace("/", "_")) / dataset_name
    fig_dir = f'{save_dir}/prediction'
    os.makedirs(fig_dir, exist_ok=True)
    n_train_examples = cfg_s.n_train_examples
    cmap = plt.get_cmap('viridis')
    n_exp = 100
    n_sample = 600
    n_layers = SELECTED_LAYERS[model_s][-1] + 1

    pcs = cfg_s.all_pc_exp
    pc_positions = np.linspace(1, 0, len(pcs))
    colors = [cmap(p) for p in pc_positions]
    pcs = [-1] + pcs  # add one for lr
    colors = ['red'] + colors  # use darkred for lr

    acc, ce_loss = get_loss_and_acc(model_s, n_layers, pcs, save_dir, n_exp=n_exp, n_sample=n_sample)
    plot_prediction_performance(acc, ce_loss, pcs, fig_dir, colors, n_sample=n_sample)
