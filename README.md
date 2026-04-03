# KG_Stock_SR_Django_Center

将原项目中 `taxwar`、`macroshock (GDP + PPI/CPI)`、`ppi` 相关页面、数据与后端逻辑集中到本目录，并使用 Django 提供服务。

当前仓库的定位是:
- 一个可运行的 Django 推理与情景分析服务。
- 一个打包后的数据与预训练权重分发目录。
- 不是完整的训练型研究仓库: 目前未包含独立训练脚本、实验配置追踪、统一评测脚本或论文级结果表。

## 目录
- `stress/`: Django app，提供页面路由和 JSON API。
- `src/models/`: PyTorch 模型定义，包含 `RAGFormer`、`LSTM`、`DTML`、`Transformer`、`MASTER`。
- `data/`: 页面与推理依赖数据。
- `saved_models/`: 已打包的模型权重。
- `scalers/`: 已打包的特征/标签标准化器。
- `templates/`: 前端页面模板。

## 环境与依赖

建议使用 Python 3.10+。

安装:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

当前 `requirements.txt` 提供运行服务所需核心依赖:
- `Django`
- `django-cors-headers`
- `numpy==1.22.4`
- `pandas==1.5.3`
- `scikit-learn==1.3.2`
- `torch`
- `joblib`

## 启动服务

```bash
python3 manage.py runserver 0.0.0.0:8000
```

页面入口:
- `/taxwar`
- `/taxwar/compare`
- `/macroshock`
- `/ppi`

## 可复现实验入口

该仓库当前可复现的是“已打包模型 + 已打包数据 + API 推理结果”，而不是“从零训练到论文结果”。

### 1. 获取历史 FSI 序列

```bash
curl http://127.0.0.1:8000/api/fsi/history
```

返回内容包含:
- `dates`
- `fsi`
- `fsi_smooth`
- `news`

### 2. 复现贸易战情景预测

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "hist_start": "2018-01",
    "short_end": "2025-12",
    "long_end": "2026-12",
    "preset": "baseline",
    "tariff": 10,
    "cny_depr": 3,
    "outflow": 500
  }'
```

关键返回字段:
- `dates_hist`, `fsi_hist`
- `dates_short`, `fsi_short`
- `dates_long`, `fsi_long`
- `short_peak`, `short_peak_date`
- `long_peak`, `long_peak_date`
- `risk_prob`, `risk_level`

### 3. 复现宏观联动情景预测

```bash
curl -X POST http://127.0.0.1:8000/api/macroshock/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "hist_start": "2018-01",
    "short_end": "2025-12",
    "long_end": "2026-12",
    "preset": "stagflation"
  }'
```

### 4. 复现 PPI 情景预测

```bash
curl -X POST http://127.0.0.1:8000/api/ppi/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "hist_start": "2015-01",
    "short_end": "2025-12",
    "long_end": "2026-12",
    "preset": "reflation"
  }'
```

### 5. 复现多模型对比

```bash
curl -X POST http://127.0.0.1:8000/api/taxwar/model-compare \
  -H 'Content-Type: application/json' \
  -d '{
    "hist_start": "2018-01",
    "short_end": "2025-12",
    "long_end": "2026-12",
    "models": ["ragformer", "lstm", "dtml", "transformer", "master"],
    "scenarios": ["baseline", "moderate", "severe", "extreme"]
  }'
```

返回中的 `metrics` 字段给出:
- `hist_mse_vs_fsi`
- `hist_mae_vs_fsi`
- `future_mse_vs_baseline`
- `future_mae_vs_baseline`
- `short_peak`, `long_peak`
- `risk_prob`, `risk_level`

## 模型与方法说明

### 数学设定

目标是基于月频宏观/金融特征序列，预测金融压力指标 `FSI_Raw`，并在给定情景冲击下生成未来月份的风险路径。

推理主流程位于 `stress/services.py`:
1. 读取 `data/taxwar/index.csv` 与 `data/fsi/fsi18_optimized.csv`。
2. 以 `Date` 对齐，构造特征矩阵 `X_t` 与目标 `y_t = FSI_Raw_t`。
3. 对输入特征与目标分别做标准化。
4. 以长度为 `seq_len` 的滑动窗口输入模型，输出下一期压力值。
5. 对 `2025-06` 之后的未来月份叠加情景扩展与启发式压力修正，得到短期/长期风险路径。

### 当前实现中的主要假设

- 数据频率为月频，且不同数据源能按月份一一对齐。
- `2025-06` 被视为历史观测区间终点，之后月份为情景外推区间。
- 若序列化的 scaler 因 NumPy 版本不兼容而无法加载，服务会退回到运行时重新拟合 scaler。
- 若模型 checkpoint 中未完整保存某些结构参数，服务会按 `stress/services.py` 中的兜底逻辑补齐。
- 未来区间的风险值不仅依赖模型原始输出，还依赖场景变量驱动的规则修正，因此未来结果不等同于“纯模型裸输出”。

### 模型复杂度

仓库当前未给出严格的理论复杂度推导。按实现可做工程级说明:
- Transformer/RAGFormer/MASTER 的注意力模块，其时间复杂度通常随序列长度近似二次增长。
- LSTM/GRU/DTML 的时序递推成本通常随序列长度近似线性增长。
- 当前数据规模较小，月频样本约 125 至 126 条，因此在线推理成本主要受模型前向传播和序列窗口遍历影响，而不是数据加载。

## 数据说明

仓库内置数据统计如下:

| File | Rows | Columns | Date Range |
| --- | ---: | ---: | --- |
| `data/fsi/fsi18_optimized.csv` | 126 | 4 | 2015-01-01 to 2025-06-01 |
| `data/ppi/index.csv` | 125 | 10 | 2015-01 to 2025-06 |
| `data/taxwar/index.csv` | 125 | 17 | 2015-01 to 2025-06 |
| `data/taxwar/policy.csv` | 125 | 6 | 2015-01 to 2025-06 |
| `data/taxwar/exchange.csv` | 125 | 6 | 2015-01 to 2025-06 |
| `data/taxwar/american.csv` | 125 | 7 | 2015-01 to 2025-06 |
| `data/taxwar/macrochina.csv` | 125 | 8 | 2015-01 to 2025-06 |
| `data/taxwar/finance.csv` | 125 | 9 | 2015-01 to 2025-06 |
| `data/taxwar/taxwar.csv` | 125 | 7 | 2015-01 to 2025-06 |

`data/taxwar/index.csv` 当前包含 16 个输入字段和 1 个日期字段:

```text
Date,Trade_Volume,Trade_Balance_China,Export_Growth_US,Import_Growth_US,GDP_Growth,Tariff_Rate,CPI,PPI,USD_CNY,LPR_SHIBOR,Stock_Volatility,Bond_Spread,Liquidity_SHIBOR,FX_Reserves_Change,Time_Trend,Trade_War_Dummy
```

`data/fsi/fsi18_optimized.csv` 当前包含:

```text
Date,FSI_Raw,FSI,News
```

### 划分方式

仓库当前没有保存标准的训练/验证/测试切分定义文件。服务中可确认的时间边界只有:
- 历史观测截止: `2025-06`
- 未来预测区间: `2025-07` 及之后

因此:
- 可复现“服务推理与情景外推”。
- 不可从仓库中严格复现“训练集/验证集/测试集”的论文级实验划分。

### 预处理

服务层已实现的预处理包括:
- 读取 CSV 后按 `Date` 对齐。
- 去除目标列与日期列后构造特征矩阵。
- `NaN` 以 `0.0` 填充后再标准化。
- 历史真实 `FSI_Raw` 在展示前执行 `ffill().bfill()`。
- 当 scaler 文件无法加载时，使用当前数据重新拟合 `StandardScaler` 作为兜底。

## 已打包模型

当前仓库包含以下权重文件:
- `saved_models/dtml_attempt_20.pth`
- `saved_models/lstm_attempt_7.pth`
- `saved_models/master_attempt_6.pth`
- `saved_models/ragformer_attempt_12.pth`
- `saved_models/ragformer_attempt_17.pth`
- `saved_models/transformer_attempt_11.pth`

服务默认优先选择同类模型中 attempt 编号最大的 checkpoint。

## Machine Learning Reproducibility Checklist Status

以下状态基于当前仓库内容，而不是理想目标。

| Checklist Item | Status | Notes |
| --- | --- | --- |
| Clear description of mathematical setting / algorithm / model | Partial | 已补充高层描述；模型定义在 `src/models/`，但尚无统一方法文档。 |
| Clear explanation of assumptions | Partial | 已在 README 说明服务层假设；尚未形成正式实验假设章节。 |
| Complexity analysis | Partial | 仅提供工程级复杂度说明，无严格推导。 |
| Clear statement of theoretical claims | No | 当前仓库未提出理论命题。 |
| Complete proof of theoretical claims | No | 当前仓库未提供。 |
| Dataset statistics | Yes | 已给出文件级样本量、列数、时间范围。 |
| Train / validation / test splits | No | 仓库未保存标准切分定义。 |
| Excluded data and preprocessing | Partial | 已说明当前服务实现中的预处理；未记录训练时排除规则。 |
| Download link to dataset / simulation environment | No | 数据已内置仓库，但未附原始来源下载链接。 |
| Data collection process for new data | No | 仓库未记录。 |
| Dependency specification | Partial | `requirements.txt` 已存在，但未完全版本锁定。 |
| Training code | No | 当前仓库未包含独立训练入口。 |
| Evaluation code | Partial | 服务中包含推理与部分误差计算逻辑，但无独立评测脚本。 |
| Pre-trained model(s) | Yes | `saved_models/` 已提供多个 checkpoint。 |
| README results table + exact commands | Partial | 已补充可执行 API 命令；尚未附固定数值结果表。 |
| Hyperparameter search range / selection / final values | No | 仓库未系统记录。 |
| Exact number of training and evaluation runs | No | attempt 文件名可见多次训练痕迹，但无正式运行记录。 |
| Definition of measures used to report results | Partial | 代码中使用 `MSE` 与 `MAE`，README 已说明相关字段。 |
| Central tendency and variation | No | 未提供均值、方差、误差条等统计。 |
| Average runtime or energy cost | No | 未提供。 |
| Computing infrastructure | Partial | 代码包含 `mps`/`cpu` 设备逻辑，但未记录实际训练硬件环境。 |

## 当前最主要的缺口

若目标是满足论文投稿或审稿中的 reproducibility checklist，至少还需要补齐:
- 独立训练脚本与统一评测脚本。
- 结果表及其对应命令。
- 训练环境、硬件、运行次数、方差统计。
- 原始数据来源与采集说明。

## 已补充的复现配置

为便于后续补训练脚本，仓库现在额外提供了:
- `configs/reproducibility.json`: 固定随机种子、时间序列切分、训练超参数、各模型默认结构参数。
- `src/reproducibility.py`: 读取配置并应用全局随机种子的工具函数。
- `src/experiment.py`: 统一的数据准备、窗口切分、训练循环、checkpoint 与 metrics 保存逻辑。
- `train.py`: 独立训练脚本。
- `evaluate.py`: 统一评测脚本。

### 固定随机种子

当前统一固定为:

```text
global_seed = 20250403
```

推荐在训练入口最开始调用:

```python
from src.reproducibility import apply_reproducibility_defaults

config = apply_reproducibility_defaults()
```

### 训练 / 验证 / 测试划分

当前采用时间序列顺序切分，不打乱:

| Split | Start | End |
| --- | --- | --- |
| Train | 2015-01 | 2022-12 |
| Validation | 2023-01 | 2024-06 |
| Test | 2024-07 | 2025-06 |

并统一使用:
- `history_window_months = 12`

这意味着:
- 训练集用于拟合模型参数。
- 验证集用于早停和超参数选择。
- 测试集仅用于最终报告。

### 超参数配置文件

`configs/reproducibility.json` 已给出:
- 通用训练配置: `optimizer`、`learning_rate`、`weight_decay`、`batch_size`、`epochs`、`patience`。
- 模型默认配置: `ragformer`、`lstm`、`dtml`、`transformer`、`master`。

## 独立训练脚本与统一评测脚本

### 训练

训练指定模型:

```bash
python3 train.py --model ragformer
```

自定义运行名:

```bash
python3 train.py --model transformer --run-name transformer_repro_v1
```

训练脚本会自动:
- 读取 `configs/reproducibility.json`
- 固定随机种子
- 按时间顺序切分 `train / validation / test`
- 以 `seq_len=12` 的滑动窗口构造样本
- 执行 early stopping
- 保存 checkpoint 到 `saved_models/`
- 保存 scaler 到 `scalers/`
- 保存 metrics 到 `artifacts/metrics/`

### 评测

评测指定 checkpoint 在测试集上的结果:

```bash
python3 evaluate.py --checkpoint saved_models/ragformer_repro_20250403.pth --split test
```

也可以评测验证集:

```bash
python3 evaluate.py --checkpoint saved_models/ragformer_repro_20250403.pth --split validation
```

### 产物格式

训练生成的 checkpoint 会尽量保持与服务推理兼容，包含:
- `state_dict`
- `model_name`
- `model_params`
- `input_size`
- `seq_len`
- `feature_columns`
- `scaler_X_path`
- `scaler_y_path`
- `metrics`

## 备注
