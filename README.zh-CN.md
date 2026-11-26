# Unibreak-Code

一个基于遗传算法（Genetic Algorithm, GA）的 **LLM 对抗攻击框架**。通过自动搜索对抗后缀 / 前缀（adversarial suffix / prefix），绕过开源与商用（API 型）大模型已有的安全对齐（safety alignment）。

> ⚠️ **仅供研究使用。** 本项目仅用于学术研究与负责任披露，用于探测 LLM 的安全对齐并推动防御研究。请勿对未获得授权的模型使用，也不得用于任何违法用途。

## 仓库结构

```
├── GA_based_methods/     # 核心 GA 攻击框架（原创）
│   ├── Encoder.py            # 对抗 token 编码与扰动采样
│   ├── Fitness_calcution.py  # 攻击成功判定 + 基于 loss 的适应度打分
│   ├── GA.py                 # GA 核心：梯度引导进化、交叉、变异
│   ├── Attack_manager.py     # 编排：模型加载、对抗库、重放与攻击
│   ├── Adv_libraray.py       # 对抗样本库（去重、top-k 复用、token 统计）
│   ├── Model_manager.py      # 开源模型 + OpenAI/DashScope 兼容 API 的封装
│   ├── Utils.py              # 配置加载
│   └── __init__.py
├── configs/               # YAML 攻击配置
├── dependency/            # 第三方 GCG 基线（llm-attacks, MIT 协议）
├── attack_demo.py         # 本地模型推理演示
├── demo.py                # API + 本地模型演示
├── replay_adv.py          # 重放已有对抗后缀
├── run_ga.py              # 单目标攻击（debug 配置）
├── run_harmful_behaviors.py  # 数据集批量攻击 + ASR 指标
├── expriments.md          # 实验 / 消融设计说明
├── requirements.txt
├── LICENSE                # MIT
└── README.md
```

## 快速开始

```bash
# 1. 安装核心依赖
pip install -r requirements.txt

# 2. 安装随仓库的 GCG 依赖（llm_attacks）
cd dependency && pip install -e . && cd ..

# 3. （可选）使用 API 型模型时，通过环境变量设置密钥。
export DASHSCOPE_API_KEY="sk-xxx"      # DashScope / 阿里云百炼
# 通过智增增访问 OpenAI 兼容代理模型（gpt/grok/doubao/...）时：
export QZZ_API_KEY="your-key"

# 4. 用 debug 配置跑单目标攻击
python run_ga.py

# 5. 在一个有害行为数据集上跑批量攻击
python run_harmful_behaviors.py --target behaviors_original --version full
```

> `run_harmful_behaviors.py` 从 `configs/attack_conf.yaml` 读取配置，并遍历
> `dependency/data/advbench/harmful_*.csv`。配置里的模型路径只是示例，请换成你自己的
> 本地 checkpoint 或 API 模型名。

## 配置说明

编辑 `configs/attack_conf.yaml`，主要字段如下：

| 字段 | 含义 |
|-----|------|
| `model_path` | 本地 checkpoint 路径，或在线模型 id |
| `model_type` | `llama-2`、`tongyi`、`qwen` 等 |
| `attack_type` | `suffix`、`prefix`、`insert`、`suffix&prefix` |
| `iters` | GA 进化代数 |
| `per_length` | 对抗 token 长度 |
| `population_size` | GA 种群大小 |
| `crossover_num` / `mutation_rate` | GA 遗传算子 |
| `use_gradient` | 是否使用梯度引导进化 |
| `adv_library` | 对抗库路径（`.pkl`），用于初始化搜索 |
| `no_ascii` | 是否将扰动限制在非 ASCII token |

## 方法思路

核心是一个**遗传算法**：把"对抗扰动"当作个体（一段 token 序列），通过
`适应度函数`评估其攻击效果，再用`梯度引导进化 / 交叉 / 变异`迭代搜索更优扰动。
`Adv_libraray` 记录历史上成功的对抗样本，作为后续搜索的初始化种子，并用于
top-k 复用与 token 敏感度统计。

- 攻击类型支持后缀（suffix）、前缀（prefix）、插入（insert）、前后缀组合。
- 攻击可从**白盒**（本地开放模型，走梯度）切换到**黑盒 / 闭源**（API 模型，走生成查询）。
- 成功判定基于模型输出是否包含"认可 + 输出"的特征词规则（`Fitness_calcution.py`）。

## 实验设计

攻击成功率先看核心指标；消融与对比实验方向见 `expriments.md`：

- **消融实验**：迭代次数、扰动强度、种群大小、是否用梯度、攻击类型（多条件组合）、
  交叉方法（改进 vs 普通）、是否初始化、是否变异。
- **对比实验**：跨模型（llama / qwen）迁移、不同扰动强度、迁移性测试。

> 说明：公开仓库只保留了攻击框架本体与可复现入口。依赖本地 `results/` 输出目录的
> 实验调度 / 数据分析脚本（`exp.py`、`data_analysis.py`、`hotmap.py`、`c-statistic.py`、
> `temp.py`、`test.py`）不在公开仓库中，仅做本地维护。

## 参考资料

随仓库的 GCG 基线：

> Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson,
> *Universal and Transferable Adversarial Attacks on Aligned Language Models*.
> <https://arxiv.org/abs/2307.15043>

## 许可协议

MIT。见 [LICENSE](./LICENSE)。随仓库的 `dependency/` 目录保留上游 `llm-attacks`
项目自身的 MIT 协议。

英文说明见 [README.md](./README.md)。