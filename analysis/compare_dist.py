import os
import joblib
import matplotlib.pyplot as plt
from utils import set_mpl, PLOT_PARAMS, load_exp_cfg
from configs.settings import SELECTED_LAYERS
from tqdm import tqdm


set_mpl()
n_bins = 25
model = 'llama3.1_8b'  # "llama3.1_8b" or "qwen2.5_7b" or "llama3.1_70b"
cfg = load_exp_cfg(model)
path = f'../results/{cfg.model_name.replace("/", "_")}/commonsense/'
layers = SELECTED_LAYERS[model]
axis = 'lr'  # 'pcascore_pc' or 'lr'
all_conditions = [''] if axis == 'lr' else cfg.all_pc_exp
save_dir = path + '/dist'
os.makedirs(save_dir, exist_ok=True)
fig_format = 'svg'
mode = 'active'


for pc in tqdm(all_conditions):
    for layer in layers:
        a = joblib.load(path+f'/hidden_last_assistant_to_eos_mean_{axis}{pc}_example_scores.pkl')
        ori_scores = a[layer]
        scores_0 = []
        scores_1 = []
        for idx in range(100):
            f_name = f'/imitate_score_by_examples_hidden_last_assistant_to_eos_mean_clf_{axis}{pc}_layer{layer}_ntrain256_mode{mode}_exp{idx}.pkl'
            b = joblib.load(path+f_name)
            scores = b['imitate_example_scores'].tolist()
            scores_0 += [scores[0], scores[3]]
            scores_1 += [scores[1], scores[2]]

        plt.figure(figsize=(2.4, 2), dpi=300)
        plt.hist(ori_scores, bins=n_bins, alpha=0.6, label='Original', density=True)
        plt.hist(scores_0, bins=n_bins, alpha=0.6, label='Imitate <0>', density=True)
        plt.hist(scores_1, bins=n_bins, alpha=0.6, label='Imitate <1>', density=True)
        plt.xlabel('Score')
        plt.ylabel('Density')
        if axis == 'lr':
            plt.title(f'LR: layer {layer+1}')
        else:
            plt.title(f'PC{pc}: layer {layer+1}')
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'pc{pc}_layer{layer}.{fig_format}'), **PLOT_PARAMS)
        plt.close()
