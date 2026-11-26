# Unibreak-Code

A genetic-algorithm (GA) based framework for adversarial attacks on aligned large
language models. It searches for adversarial suffixes / prefixes that bypass the
safety alignment of both open-source and commercial (API-based) models.

> ⚠️ **Research purpose only.** This project is published for academic research and
> responsible disclosure. The methods here can be used to probe safety alignment of
> LLMs and to improve defence research. Do not use it against models you are not
> authorised to test, or for unlawful purposes.

## What this repo contains

- `GA_based_methods/` — the core GA attack framework (original work):
  - `Encoder.py` — adversarial token encoding & perturbation sampling
  - `Fitness_calcution.py` — attack success detection + loss-based fitness scoring
  - `GA.py` — genetic algorithm core (gradient-guided evolution, crossover, mutation)
  - `Attack_manager.py` — orchestration: model load, adversarial library, replay & attack
  - `Adv_libraray.py` — adversarial-sample library (dedup, top-k retrieval, token stats)
  - `Model_manager.py` — model wrapper for open-source & OpenAI/DashScope-compatible APIs
  - `Utils.py` — config loading
- `configs/` — YAML attack configuration files
- Dependency (third-party, under `dependency/`): the [llm-attacks / GCG](https://github.com/llm-attacks/llm-attacks) baseline
  (MIT license), including `minimal_gcg` used at runtime by `Attack_manager`.
- Top-level scripts — runnable demos & experiments (see below).

## Quick start

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Install the bundled GCG dependency (llm_attacks)
cd dependency && pip install -e . && cd ..

# 3. (Optional) If you use API-based models, set your keys as environment variables.
export DASHSCOPE_API_KEY="sk-xxx"      # DashScope / Aliyun Bailian
# For OpenAI-compatible proxy models (gpt/grok/doubao/...) via zhizengzeng:
export QZZ_API_KEY="your-key"

# 4. Run a single-target attack with the debug config
python run_ga.py

# 5. Run a batch over a harmful-behavior dataset
python run_harmful_behaviors.py --target behaviors_original --version full
```

> `run_harmful_behaviors.py` reads attack configs from `configs/attack_conf.yaml` and
> iterates over `dependency/data/advbench/harmful_*.csv`. Model paths in the configs
> are examples; point them to your own local checkpoints or API model names.

## Configuration

Edit `configs/attack_conf.yaml`. Key fields:

| Key | Meaning |
|-----|---------|
| `model_path` | Local checkpoint path or online model id |
| `model_type` | `llama-2`, `tongyi`, `qwen`, ... |
| `attack_type` | `suffix`, `prefix`, `insert`, `suffix&prefix` |
| `iters` | Number of GA generations |
| `per_length` | Adversarial-token length |
| `population_size` | GA population size |
| `crossover_num` / `mutation_rate` | GA genetic operators |
| `use_gradient` | Use gradient-guided evolution |
| `adv_library` | Path to adversarial library (`.pkl`) to seed the search |
| `no_ascii` | Restrict perturbations to non-ASCII tokens |

## Top-level scripts

- `run_ga.py` — single-target attack using `configs/attack_conf_debug.yaml`.
- `run_harmful_behaviors.py` — batch attack over an AdvBench-style dataset, with ASR / avg-iteration metrics.
- `attack_demo.py` / `demo.py` — small demonstrations of local model inference & API calls.
- `replay_adv.py` — replay an existing adversarial suffix against a model.
- `exp.py`, `data_analysis.py`, `hotmap.py`, `temp.py`, `c-statistic.py` — experiment
  tooling / result analysis (require local `results/` output and optional datasets).
- `test.py` — CSV preprocessing helper for a benchmark dataset.

## Reference

The bundled GCG baseline:

> Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson,
> *Universal and Transferable Adversarial Attacks on Aligned Language Models*.
> <https://arxiv.org/abs/2307.15043>

## License

MIT. See [LICENSE](./LICENSE). The bundled `dependency/` directory retains its own
MIT license from the upstream `llm-attacks` project.