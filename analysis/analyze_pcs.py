from joblib import load
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from pathlib import Path
from utils import set_mpl, PLOT_PARAMS


def explained_variance_on_direction(X: np.ndarray, w: np.ndarray):
    X_centered = X - X.mean(axis=0)
    projection = X_centered @ w
    return np.var(projection)

fig_format = 'svg'
set_mpl()
cmap = cm.get_cmap('viridis')
pcs = [1, 2, 4, 8, 32, 128, 512]
pc_positions = np.linspace(1, 0, len(pcs))
colors = [cmap(p) for p in pc_positions]

model_name = "meta-llama_Llama-3.1-8B-Instruct"
dataset_name = "commonsense"
save_dir = Path("../results") / model_name / dataset_name
data_dict = load(save_dir / f"hidden_last_assistant_to_eos_mean_data_Xy.pkl")
train_X, train_y = data_dict["train"]["X"], data_dict["train"]["y"]


explained_variances = []
for layer, X in train_X.items():
    X -= X.mean(axis=0)
    pca = PCA(n_components=np.min(X.shape)-1)
    pca.fit(X)
    explained_variance = pca.explained_variance_ratio_
    explained_variances.append(explained_variance)
explained_variances = np.array(explained_variances) # shape: [n_layers, n_components]
explained_variance_mean = np.mean(explained_variances, axis=0)


lr_clf = load(save_dir / f"hidden_last_assistant_to_eos_mean_classifiers_lr.pkl")
pca_clf = load(save_dir / f"hidden_last_assistant_to_eos_mean_classifiers_pcascore_pc1.pkl")

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
