# Unibreak-Code

A genetic-algorithm (GA) based framework for adversarial attacks against aligned
large language models. It searches for adversarial suffixes / prefixes that bypass
the safety alignment of both open-source and commercial (API-based) models.

> ⚠️ **Research purpose only.** Published for academic research and responsible
> disclosure. Use it only to probe safety alignment of LLMs and to improve defence
> research. Do not use it against models you are not authorised to test, or for any
> unlawful purpose.

## Repository layout

```
├── GA_based_methods/     # Core GA attack framework (original work)
│   ├── Encoder.py            # Adversarial token encoding & perturbation sampling
│   ├── Fitness_calcution.py  # Attack-success detection + loss-based fitness scoring
│   ├── GA.py                 # GA core: gradient-guided evolution, crossover, mutation
│   ├── Attack_manager.py     # Orchestration: model load, library, replay & attack
│   ├── Adv_libraray.py       # Adversarial-sample library (dedup, top-k, token stats)
│   ├── Model_manager.py      # Model wrapper for open-source & OpenAI/DashScope APIs
│   ├── Utils.py              # Config loading
│   └── __init__.py
├── configs/               # YAML attack configurations
├── dependency/            # Third-party GCG baseline (llm-attacks, MIT)
├── attack_demo.py         # Local model inference demo
├── demo.py                # API + local model demo
├── replay_adv.py          # Replay an existing adversarial suffix
├── run_ga.py              # Single-target attack (debug config)
├── run_harmful_behaviors.py  # Batch attack over a dataset + ASR metrics
├── expriments.md          # Experiment / ablation design notes
├── requirements.txt
├── LICENSE                # MIT
└── README.md
```

## Quick start

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Install the bundled GCG dependency (llm_attacks)
cd dependency && pip install -e . && cd ..

# 3. (Optional) For API-based models, set keys via environment variables.
export DASHSCOPE_API_KEY="sk-xxx"      # DashScope / Aliyun Bailian
# For OpenAI-compatible proxy models (gpt/grok/doubao/...) via zhizengzeng:
export QZZ_API_KEY="your-key"

# 4. Run a single-target attack with the debug config
python run_ga.py

# 5. Run a batch over a harmful-behavior dataset
python run_harmful_behaviors.py --target behaviors_original --version full
```

> `run_harmful_behaviors.py` reads configs from `configs/attack_conf.yaml` and iterates
> over `dependency/data/advbench/harmful_*.csv`. Model paths in the configs are
> examples; point them to your own local checkpoints or API model names.

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

## Reference

The bundled GCG baseline:

> Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson,
> *Universal and Transferable Adversarial Attacks on Aligned Language Models*.
> <https://arxiv.org/abs/2307.15043>

## License

MIT. See [LICENSE](./LICENSE). The bundled `dependency/` directory retains its own
MIT license from the upstream `llm-attacks` project.

For Chinese documentation, see [README.zh-CN.md](./README.zh-CN.md).