from joblib import load
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cosine
from analysis.process_hidden import eval_classify_hiddens
from pathlib import Path
from analysis.stats_fn import mean_ci
from utils import set_mpl, PLOT_PARAMS, safe_dump, load_exp_cfg


def explained_variance_on_direction(X: np.ndarray, w: np.ndarray):
    X_centered = X - X.mean(axis=0)
    projection = X_centered @ w
    return np.var(projection)


def calc_explained_variance(train_X):
    explained_variances = []
    for layer, X in train_X.items():
        X -= X.mean(axis=0)
        pca = PCA(n_components=np.min(X.shape)-1)
        pca.fit(X)
        explained_variance = pca.explained_variance_ratio_
        explained_variances.append(explained_variance)
    explained_variances = np.array(explained_variances) # shape: [n_layers, n_components]
    explained_variance_mean = np.mean(explained_variances, axis=0)
    return explained_variance_mean


def plot_explained_variance(explained_variance_mean, lr_explained_variance_mean, pcs, colors, save_dir):
    diff = np.abs(explained_variance_mean - lr_explained_variance_mean)
    min_idx = np.argmin(diff)
    x_cross = min_idx + 1
    y_cross = explained_variance_mean[min_idx]

    plt.figure(figsize=(2, 1.75), dpi=300)
    plt.loglog(range(1, len(explained_variance_mean) + 1), explained_variance_mean, color='grey')
    for pc_number, color in zip(pcs,colors):
        plt.plot([pc_number, pc_number], [explained_variance_mean[-1], explained_variance_mean[pc_number-1]], '--',
                 label=f'PC {pc_number}',color=color, linewidth=1.5)
    plt.plot([x_cross], [y_cross], 'x', color='red', label='LR', markersize=6)
    plt.xlim(0.9, len(explained_variance_mean))
    plt.ylim(1e-4, 0.1)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"imitation/hidden_explained_variance.{fig_format}", **PLOT_PARAMS)
    plt.show()
    plt.close()


def plot_lr_pc_axis_similarity(lr_pca_similarity, pcs, colors, save_dir):
    plt.figure(figsize=(2, 1.75), dpi=300)
    for i, pc in enumerate(pcs):
        plt.plot(range(len(lr_clf)), np.abs(lr_pca_similarity[:, i]), label=f'PC{pc}',
                 color=colors[i], linewidth=1, alpha=0.7)
    plt.legend(fontsize=5, loc='upper right', bbox_to_anchor=(1.05, 1.1))
    plt.xticks([0, 15, 31], ['1', '16', '32'])
    plt.xlabel('Layer')
    plt.ylabel('Similarity: LR axis vs. PCs', fontsize=7)
    plt.tight_layout()
    plt.savefig(save_dir / f"imitation/lr_pca_similarity.{fig_format}", **PLOT_PARAMS)
    plt.close()


def plot_layers_pc_lr_classifier_performance(data_dict, lr_clf, pca_clf, pcs, save_dir):
    train_X, train_y = data_dict["train"]["X"], data_dict["train"]["y"]
    test_X, test_y = data_dict["test"]["X"], data_dict["test"]["y"]

    n_cols = 4
    fig, ax = plt.subplots(2, n_cols, figsize=(6, 3), dpi=300, sharey=True, sharex=True)
    for i, pc_number in enumerate(pcs):
        for k in pca_clf.keys():
            pca_clf[k].pc_number = pc_number
        all_train_accuracies = eval_classify_hiddens(train_X, train_y, pca_clf, return_type='accuracy')
        all_test_accuracies = eval_classify_hiddens(test_X, test_y, pca_clf, return_type='accuracy')
        ax[i // n_cols, i % n_cols].plot(list(all_train_accuracies.keys()), list(all_train_accuracies.values()), label="Train")
        ax[i // n_cols, i % n_cols].plot(list(all_test_accuracies.keys()), list(all_test_accuracies.values()), label="Test")
        ax[i // n_cols, i % n_cols].axhline(y=0.5, color='grey', linestyle='--')  # chance level
        ax[i // n_cols, i % n_cols].set_xlabel("Layer")
        ax[i // n_cols, i % n_cols].set_title(f"PC{pc_number}")

    all_train_accuracies = eval_classify_hiddens(train_X, train_y, lr_clf, return_type='accuracy')
    all_test_accuracies = eval_classify_hiddens(test_X, test_y, lr_clf, return_type='accuracy')
    ax[-1, -1].plot(list(all_train_accuracies.keys()), list(all_train_accuracies.values()), label="Train")
    ax[-1, -1].plot(list(all_test_accuracies.keys()), list(all_test_accuracies.values()), label="Test")
    ax[-1, -1].axhline(y=0.5, color='grey', linestyle='--')  # chance level
    ax[-1, -1].set_xlabel("Layer")
    ax[-1, -1].set_title("LR")

    for i in range(2):
        ax[i, 0].set_ylabel("Accuracy")
    ax[0, 0].legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"imitation/classifier_performance.{fig_format}", **PLOT_PARAMS)
    plt.close()


def ideal_observer_performance(axis_clf, train_X, test_X, layer):
    X = train_X[layer]
    nf_y = axis_clf.predict(X)
    x_test = test_X[layer]
    y_test = axis_clf.predict(x_test)
    ideal = LogisticRegression(solver='saga', max_iter=1000, random_state=42, class_weight='balanced', n_jobs=-1)
    ideal.fit(X, nf_y)
    acc = ideal.score(x_test, y_test)
    return acc


def calc_ideal_observer_performance(lr_clf, pca_clf, pcs, train_X, test_X, save_dir):
    layers_acc = defaultdict(list)

    for pc in tqdm(pcs):
        for layer, pc_clf in pca_clf.items():
            pc_clf.pc_number = pc
            acc = ideal_observer_performance(pc_clf, train_X, test_X, layer)
            layers_acc[pc].append(acc)

    for layer, clf in lr_clf.items():
        acc = ideal_observer_performance(clf, train_X, test_X, layer)
        layers_acc['lr'].append(acc)

    safe_dump(layers_acc, save_dir / f"imitation/layers_acc_ideal_observer.pkl")
    return layers_acc


def plot_ideal_observer_performance(layers_acc, pcs, colors, save_dir):
    plt.figure(figsize=(3.6, 3), dpi=300)
    plt.plot(range(len(layers_acc['lr'])), layers_acc['lr'], label='LR', color='red', linewidth=1.5)

    for i, pc in enumerate(pcs):
        plt.plot(range(len(layers_acc[pc])), layers_acc[pc], label=f'PC{pc}', color=colors[i], linewidth=1.5)

    # plt.axhline(y=0.5, color='grey', linestyle='--')  # chance level
    plt.ylim(0.5, 1)
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(save_dir / f"imitation/ideal_perf_layers.{fig_format}", **PLOT_PARAMS)
    plt.close()

def plot_models_ideal_observer_performance(dataset_name, pcs):
    model_size = 3
    red_cmap = plt.get_cmap('Reds_r')
    reds = [red_cmap(i / model_size) for i in range(model_size)]
    blue_cmap = plt.get_cmap('Blues_r')
    blues = [blue_cmap(i / model_size) for i in range(model_size)]
    colors = {'llama3.1_8b': reds[0], 'llama3.2_3b': reds[1], 'llama3.2_1b': reds[2],
              'qwen2.5_7b': blues[0], 'qwen2.5_3b': blues[1], 'qwen2.5_1.5b': blues[2]}

    plt.figure(figsize=(2.5, 2), dpi=300)
    for model in ["llama3.1_8b", "llama3.2_3b", "llama3.2_1b", "qwen2.5_7b", "qwen2.5_3b", "qwen2.5_1.5b"]:
        cfg = load_exp_cfg(model)
        model_name = cfg.model_name.replace("/", "_")
        layers_acc = load(f'../results/{model_name}/{dataset_name}/imitation/layers_acc_ideal_observer.pkl')
        acc_mean = []
        acc_ci = []
        mean, ci = mean_ci(np.array(layers_acc['lr']), axis=0)
        acc_mean.append(mean)
        acc_ci.append(ci)
        for pc in pcs:
            mean, ci = mean_ci(np.array(layers_acc[pc]), axis=0)
            acc_mean.append(mean)
            acc_ci.append(ci)

        acc_mean = np.array(acc_mean)
        acc_ci = np.array(acc_ci)
        plt.plot(acc_mean, color=colors[model], alpha=0.7, label=model)
        # plt.fill_between(range(len(acc_mean)), acc_mean - acc_ci, acc_mean + acc_ci, alpha=0.2)
    plt.ylim(0.5, 1)
    plt.legend(fontsize=6)
    plt.xlabel("Target Axis")
    plt.xticks(range(len(pcs) + 1), ['LR'] + [f'PC{pc}' for pc in pcs], fontsize=5.5)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(f'../results/ideal_perf_layers_models.svg', **PLOT_PARAMS)
    plt.close()


if __name__ == "__main__":
    fig_format = 'svg'
    set_mpl()
    cmap = plt.get_cmap('viridis')
    pcs = [1, 2, 4, 8, 32, 128, 512]
    pc_positions = np.linspace(1, 0, len(pcs))
    colors = [cmap(p) for p in pc_positions]

    model_name = "meta-llama_Llama-3.1-8B-Instruct"
    dataset_name = "commonsense"
    save_dir = Path("../results") / model_name / dataset_name
    plot_models_ideal_observer_performance(dataset_name, pcs)

    data_dict = load(save_dir / f"hidden_last_assistant_to_eos_mean_data_Xy.pkl")
    train_X, train_y = data_dict["train"]["X"], data_dict["train"]["y"]
    test_X, test_y = data_dict["test"]["X"], data_dict["test"]["y"]
    lr_clf = load(save_dir / f"hidden_last_assistant_to_eos_mean_classifiers_lr.pkl")
    pca_clf = load(save_dir / f"hidden_last_assistant_to_eos_mean_classifiers_pcascore_pc1.pkl")

    layers_acc = calc_ideal_observer_performance(lr_clf, pca_clf, pcs, train_X, test_X, save_dir)
    # layers_acc = load(save_dir / f"imitation/layers_acc_ideal_observer.pkl")
    plot_ideal_observer_performance(layers_acc, pcs, colors, save_dir)

    lr_explained_variance = []
    lr_pca_similarity = np.zeros((len(lr_clf), len(pcs))) # shape: [n_layers, n_components]

    for layer, clf in lr_clf.items():
        X = train_X[layer]
        w = clf.coef_.squeeze()
        w /= np.linalg.norm(w)
        explained_variance = explained_variance_on_direction(X, w)
        total_variance = np.var(X, axis=0).sum()
        lr_explained_variance.append(explained_variance / total_variance)

        for i, pc in enumerate(pcs):
            layer_pca_clf = pca_clf[layer]
            layer_pca_clf.pc_number = pc
            lr_pca_similarity[layer, i] = 1 - cosine(w, layer_pca_clf.axis_)

    lr_explained_variance = np.array(lr_explained_variance) # shape: [n_layers, n_components]
    lr_explained_variance_mean = np.mean(lr_explained_variance, axis=0)

    plot_explained_variance(explained_variance_mean, lr_explained_variance_mean, pcs, colors, save_dir)
    plot_lr_pc_axis_similarity(lr_pca_similarity, pcs, colors, save_dir)
    plot_layers_pc_lr_classifier_performance(data_dict, lr_clf, pca_clf, pcs, save_dir)
