# LLM Neurofeedback

Official code repository for [Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations](https://arxiv.org/abs/2505.13763).

## Project Overview
- Neurofeedback experiment (`neurofeedback.py`) is the primary study pipeline, running active trials (LLM-generated sentences) and inactive trials (provided sentences) to steer activations through neurofeedback.
- Hidden-state preparation (`main_prep.py`) extracts model activations for a dataset, trains hidden-state classifiers, and saves caches under `results/<model>/<dataset>/`.
- Control experiment (`main_imitexp.py`) replays PC or LR axes to verify that activation steering changes behavior as expected.
- Report experiment (`main_predexp.py`) evaluates how well scores from one configuration predict outcomes in another (cross-model or cross-setting).
- Plotting helpers (`plot_imit.py`, `plot_pred.py`) consolidate metrics, while analysis utilities in `analysis/` implement the feature extraction, classifier training, statistics, and figure generation used across the entry points.

## Repository Layout
- `analysis/` - Core analysis modules such as `process_hidden.py` (hidden-state extraction), `classifiers.py` (LR/PC models), `process_prompts.py` (prompt tagging), and `stats_fn.py`.
- `configs/` - YAML settings for prompts (`meta_prompts.yml`), dataset tags (`prompt_tags.yml`), analysis defaults (`analysis.yml`), and Python helpers (`settings.py`, `nf_exp1.yml`). Update these before modifying code.
- `data/` - Read-only datasets (commonsense, true_false, sycophancy, emotion). Place new assets in `data/custom/` with a README.
- `main_prep.py`, `main_imitexp.py`, `main_predexp.py` - Command-line entry points for the three experiment stages.
- `neurofeedback.py` - Main neurofeedback experiment orchestrating active/inactive trials and logging behavioral changes.
- `plot_imit.py`, `plot_pred.py` - Convenience scripts for visual summaries of cached experiments.
- `utils.py` - Shared helpers for seeding, model loading, configuration aliases, and serialization.
- `submit.slurm`, `submit_imit.slurm`, `submit_pred.slurm` - Slurm templates for batch execution (preparation, imitation control, and prediction).
- `tasks/` - Slot for task-specific scripts or notebooks (populate as needed).
- `results/` - Auto-created output directory (`results/<model>/<dataset>/`) containing `.pkl` caches and figures.

## Environment Setup
```bash
uv venv
uv pip install -r requirements.txt
```
You can also skip activation and prefix commands with `uv run`, e.g., `uv run python main_prep.py …`.

## Configuring Experiments
- Align dataset, prompt, and classifier settings in `configs/nf_exp1.yml`, `configs/meta_prompts.yml`, and `configs/prompt_tags.yml` before running.
- Adjust layer subsets in `configs/settings.py::SELECTED_LAYERS`.
- Supported dataset keys: `commonsense`, `true_false`, `sycophancy`, `emotion`.
- The paper reports results using PC axes `[1, 2, 4, 8, 32, 128, 512]` and an LR classifier (`--clf lr`) for comparison; replicate runs should include these settings.
- Model aliases like `llama3_8b` are expanded in `utils.py` (e.g., to `meta-llama/Meta-Llama-3.1-8B-Instruct`); full Hugging Face names are also accepted.
- Generated artifacts (hidden-state caches, classifier pickles, plots) land in `results/<model>/<dataset>/`. Avoid committing these artifacts.

## Running Locally (Python)
Activate your environment and run the three stages:

```bash
# 1. Prepare hidden-state caches
python main_prep.py --model llama3_8b --dataset commonsense --process_hidden_method pcascore

# 2. Run imitation / control experiments
python main_imitexp.py --model llama3_8b --dataset commonsense --pc 1 --imit_exp_end -1
# or logistic regression control
python main_imitexp.py --model llama3_8b --dataset commonsense --clf lr --imit_exp_end -1

# 3. Cross-model predictions (scores from config_s replayed on config_e)
python main_predexp.py --config_s llama3_8b --config_e qwen2.5_7b --dataset commonsense --pred_exp_end -1
```

Use `--*_end -1` to process the full dataset; omit or change the value for quick smoke tests. Inspect the corresponding directory in `results/` to verify `.pkl` caches and `.png/.pdf` plots.

## Running on Slurm
1. Review `submit.slurm`, `submit_imit.slurm`, and `submit_pred.slurm` before submission:
   - Update `#SBATCH` directives (`--account`, `-q/--partition`, `--gres`, `--mem-per-gpu`, `--time`, log output path).
   - Ensure `module load` and `conda activate` lines match your cluster environment; point to the environment created above.
   - Create the log directory referenced by `#SBATCH -o` (e.g., `mkdir -p temp`) or change it.
   - Uncomment or swap the `python main_*.py` invocation if you adapt a template for another stage.
2. Submit jobs with the same flags used locally. Examples:
   ```bash
   sbatch submit.slurm --model llama3_8b --dataset commonsense
   sbatch submit_imit.slurm --model llama3_8b --dataset commonsense --pc 1
   sbatch submit_pred.slurm --config_s llama3_8b --config_e qwen2.5_7b --dataset commonsense --pc 1
   ```
3. Monitor logs under the path supplied to `#SBATCH -o` and confirm `results/<model>/<dataset>/` fills with the expected artifacts.

## Outputs and Next Steps
- Hidden-state arrays, classifier checkpoints, and score tables are stored as `.pkl` files; plots are saved as `.png`/`.pdf`.
- Use `plot_imit.py` and `plot_pred.py` to regenerate figures from cached runs.
- For new datasets, add a README under `data/custom/<dataset_name>/` documenting provenance and preprocessing.
- Validate code changes by running the smallest representative experiment (`--*_end -1`) and reviewing updated artifacts in `results/`.
