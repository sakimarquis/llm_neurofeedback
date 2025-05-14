from pathlib import Path
import numpy as np
from joblib import load
from utils import load_exp_cfg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


def get_nf_optimal_perf(x, y, n_layers, n_samples):
    clf = LogisticRegression()

    all_nll = []
    all_acc = []
    for i in range(n_layers):
        clf.fit(x[i][:n_samples], y[:n_samples])

        if n_samples < 200:
            test_x = x[i][n_samples:]
            test_y = y[n_samples:]
        else:
            test_x = x[i]
            test_y = y
        acc = clf.score(test_x, test_y)
        probs = clf.predict_proba(test_x)
        nll = log_loss(test_y, probs, labels=clf.classes_, normalize=True)
        all_acc.append(acc)
        all_nll.append(nll)

    return all_nll, all_acc


if __name__ == "__main__":
    cfg = load_exp_cfg('llama3.1_70b')
    file_name = f'{cfg.clf}_{cfg.clf_name}'
    dataset_name, label_name = "happy_sad", "labels"
    save_dir = Path("../results") / (cfg.model_name.replace("/", "_")) / dataset_name
    fname = f"hidden_last_assistant_to_eos_mean_data_Xy.pkl"

    train = load(save_dir / fname)['train']
    x = train['X']
    y = np.array(train['y'])

    all_nll, all_acc = get_nf_optimal_perf(x, y, n_layers=80, n_samples=30)
    plt.plot(all_nll, label='nll')
    plt.show()
