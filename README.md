# BERT Fine-Tuning on GLUE Benchmark

基于 BERT-mini 的 GLUE 基准微调：MRPC 句子对语义等价判断 + SST-2 情感分类

Fine-tuning BERT-mini on GLUE benchmark: MRPC paraphrase detection + SST-2 sentiment classification

---

## 目录 / Table of Contents

- [项目简介 / Overview](#项目简介--overview)
- [模型架构 / Model Architecture](#模型架构--model-architecture)
- [数据集 / Datasets](#数据集--datasets)
- [训练配置 / Training Configuration](#训练配置--training-configuration)
- [项目结构 / Project Structure](#项目结构--project-structure)
- [快速开始 / Quick Start](#快速开始--quick-start)
- [使用方法 / Usage](#使用方法--usage)

---

## 项目简介 / Overview

**中文**：本项目使用轻量级预训练模型 `prajjwal1/bert-mini`（4层 Transformer Encoder），在 GLUE 基准的两个任务上进行微调：
- **MRPC**（Microsoft Research Paraphrase Corpus）：判断两个句子是否语义等价
- **SST-2**（Stanford Sentiment Treebank）：判断句子情感极性（正面/负面）

项目基于 HuggingFace Transformers 的 `Trainer` API 实现，包含完整的训练、评估、推理和可视化流程。

**English**: This project fine-tunes a lightweight pre-trained model `prajjwal1/bert-mini` (4-layer Transformer Encoder) on two GLUE benchmark tasks:
- **MRPC**: Determine whether two sentences are semantically equivalent
- **SST-2**: Classify sentence sentiment polarity (positive/negative)

Built on HuggingFace Transformers `Trainer` API with complete training, evaluation, inference, and visualization pipelines.

---

## 模型架构 / Model Architecture

| 参数 / Parameter | 值 / Value |
|---|---|
| 预训练模型 / Pre-trained Model | `prajjwal1/bert-mini` |
| 隐藏层维度 / Hidden Size | 256 |
| Encoder 层数 / Num Layers | 4 |
| 注意力头数 / Attention Heads | 4 |
| 词表大小 / Vocab Size | 30,522 |
| 分词器 / Tokenizer | `bert-base-uncased` (WordPiece) |
| 分类头 / Classification Head | Linear(256, 2) |

### 微调策略 / Fine-tuning Strategy

```
[CLS] + Sentence(s) + [SEP] → BERT-mini Encoder (4 layers) → [CLS] hidden state → Linear → Softmax → Label
```

- 全参数微调（所有 Encoder 层 + 分类头）
- AdamW 优化器，学习率 3e-5
- SST-2 使用 warmup_ratio=0.1，MRPC 不使用 warmup

---

## 数据集 / Datasets

| 任务 / Task | 训练集 / Train | 验证集 / Validation | 输入格式 / Input | 标签 / Labels |
|---|---|---|---|---|
| MRPC | 3,668 | 408 | 句子对 / Sentence pair | 0=不等价, 1=等价 |
| SST-2 | 67,349 | 872 | 单句 / Single sentence | 0=负面, 1=正面 |

数据集通过 HuggingFace `datasets` 库自动下载（GLUE benchmark）。

---

## 训练配置 / Training Configuration

| 参数 / Parameter | MRPC | SST-2 |
|---|---|---|
| 学习率 / Learning Rate | 3e-5 | 3e-5 |
| Batch Size | 32 | 32 |
| Epochs | 2 | 2 |
| Max Length | 100 | 64 |
| Warmup Ratio | 0 | 0.1 |
| 评估指标 / Metrics | Accuracy, F1 | Accuracy |
| 早停 / Early Stopping | 基于 best accuracy | 基于 best accuracy |

---

## 项目结构 / Project Structure

```
.
├── main.py                          # 主程序入口 / Main entry point
├── requirements.txt                 # 依赖 / Dependencies
├── src/
│   ├── trainers/
│   │   └── bert_trainer.py          # BERT 微调训练器 / Fine-tuning trainer
│   ├── inference/
│   │   └── predictor.py             # 推理预测器 / Inference predictor
│   └── official_code/               # 参考代码 / Reference code
│       ├── mrpc_1.py
│       └── sst2_1.py
├── results/                         # 实验结果（不上传）/ Results (not tracked)
│   └── {timestamp}_{task}/
│       ├── model/                   # 微调后的模型 / Fine-tuned model
│       ├── checkpoints/             # 训练检查点 / Checkpoints
│       ├── logs/                    # TensorBoard 日志 / TB logs
│       ├── experiment_results.json  # 完整实验数据 / Full results
│       ├── summary.json             # 结果摘要 / Summary
│       └── training_curves.png      # 训练曲线 / Training curves
└── data/                            # 数据缓存 / Data cache
```

---

## 快速开始 / Quick Start

### 环境安装 / Installation

```bash
pip install -r requirements.txt
```

主要依赖 / Key dependencies:
- `transformers >= 4.30.0`
- `datasets`
- `evaluate`
- `torch >= 2.0`
- `matplotlib`

### 一键训练 / Train All

```bash
python main.py --mode all
```

---

## 使用方法 / Usage

### 交互式模式 / Interactive Mode

```bash
python main.py
```

提供菜单选择：训练 MRPC / SST-2 / 全部训练 / 推理预测。

### 命令行模式 / CLI Mode

**训练单个任务 / Train single task:**
```bash
python main.py --mode train --task mrpc
python main.py --mode train --task sst2
```

**自定义参数 / Custom parameters:**
```bash
python main.py --mode train --task mrpc --lr 3e-5 --batch_size 32 --epochs 2 --seed 42
```

**推理预测 / Inference:**
```bash
python main.py --mode predict --task sst2
```

### 参数说明 / Arguments

| 参数 / Arg | 说明 / Description | 默认值 / Default |
|---|---|---|
| `--mode` | 运行模式：train/predict/all/interactive | interactive |
| `--task` | 任务：mrpc/sst2 | - |
| `--lr` | 学习率 / Learning rate | 3e-5 |
| `--batch_size` | 批次大小 / Batch size | 32 |
| `--epochs` | 训练轮数 / Epochs | 2 |
| `--seed` | 随机种子 / Random seed | 42 |

---

## 实验输出 / Experiment Output

每次训练自动生成以下文件 / Each training run generates:

| 文件 / File | 说明 / Description |
|---|---|
| `experiment_results.json` | 完整实验数据（配置、训练历史、评估指标、预测样本） |
| `summary.json` | 结果摘要 |
| `training_curves.png` | 训练曲线（Loss、学习率、评估指标、每轮耗时） |
| `model/` | 微调后的 BERT 模型（可直接加载推理） |
| `checkpoints/` | 训练检查点 |
| `logs/` | TensorBoard 日志 |

---

## 功能特性 / Features

- 模块化设计：训练器（`BertTrainer`）和预测器（`BertPredictor`）解耦，支持独立使用
- 实验管理：每次实验自动生成唯一 ID，所有产物（模型、日志、图表、结果）统一保存
- 训练监控：自定义 `TrainerCallback` 记录每步 loss、学习率、评估指标和耗时
- 可视化：自动生成 4 子图训练曲线（Loss / LR / Metrics / Time）
- 推理支持：交互式预测 + 批量预测 + 随机样本预测展示
