# 项目说明

本仓库用于 EEG 分类任务，当前包含 `BCIC2A`、`CHINESE`、`MDD`、`SEED`、`SLEEP` 等数据集的训练与推理代码。

如果要训练新的模型，建议按下面的方式添加：

- 新建对应模型文件，命名为 `<dataset>_<model>.py`
- 在 `configs/<dataset>/<model>.yaml` 下添加该模型对应的配置文件
- 再将新模型接入训练入口中的模型构建逻辑

## 项目入口

项目主入口是 `train.py`。

- `train.py`：训练模型，并保存 checkpoint 和训练历史。
- `test.py`：加载训练好的 checkpoint，在测试集上生成预测结果。
- `train.ipynb`：Notebook 版本，适合交互式实验与调试，不是主要命令行入口。

## 基本流程

1. 按要求准备数据集目录。
2. 运行 `train.py` 训练模型。
3. 运行 `test.py` 加载 checkpoint 并生成提交结果。

示例：

```bash
python3 train.py --dataset MDD --model lstm
python3 test.py --dataset MDD --checkpoint checkpoints/mdd_lstm.pt
```

如果训练睡眠任务的 CTNet：

```bash
python3 train.py --dataset SLEEP --model sleep_ctnet
```

## 配置文件

训练支持从 `configs/` 目录读取 YAML 配置。

- `configs/default.yaml`：通用默认超参数配置。
- `configs/sleep/ctnet.yaml`：`SLEEP + sleep_ctnet` 的专用超参数配置。

`train.py` 中的默认配置加载规则如下：

- 当 `--dataset SLEEP` 且 `--model sleep_ctnet` 时，默认加载 `configs/sleep/ctnet.yaml`。
- 其他情况默认加载 `configs/default.yaml`。
- 如果显式传入 `--config`，则优先使用指定配置文件。
- 这些 YAML 主要用于保存超参数，不负责决定数据集和模型类型。

## 数据目录结构

每个数据集目录下默认应包含：

- `train.h5`：训练集，包含 `X` 和 `y`
- `val.h5`：验证集，包含 `X` 和 `y`
- `test_x_only.h5`：测试集，只包含 `X`
- `dataset_info.json` 或 `dataset_info_fixed.json`：数据集元信息，例如通道列表和类别列表

示例：

```text
course project/
  MDD/
    train.h5
    val.h5
    test_x_only.h5
    dataset_info.json
```

## 文件说明

### 主脚本

- `train.py`：训练入口。负责解析命令行参数、加载 YAML 配置、读取数据集信息、构建模型、训练、验证并保存结果。
- `test.py`：推理入口。负责读取 `train.py` 生成的 checkpoint，重建模型，并在测试集上输出预测标签文本。
- `train.ipynb`：交互式实验 Notebook，可用于快速试验参数、检查数据、调试训练流程。

### 模型与流程相关文件

- `eeg_pipeline.py`：训练和推理共用的工具函数，包括随机种子设置、数据集路径解析、数据集信息读取、设备选择和模型构建。
- `RNN.py`：RNN 基线模型实现，包含 `EEGLSTM` 和 `EEGGRU`。
- `sleep_ctnet.py`：睡眠任务使用的 CTNet 模型实现，包含卷积 patch embedding 和 Transformer 编码器分类头。

### 数据读取相关文件

- `TEST_DATASET.py`：PyTorch 数据集封装。`TrainDataset` 读取训练/验证用的 `X` 与 `y`，`TestDataset` 只读取测试用的 `X`。

### 配置文件

- `configs/default.yaml`：通用训练超参数。
- `configs/sleep/ctnet.yaml`：`SLEEP` 数据集配合 `sleep_ctnet` 时使用的超参数。

## 输出文件

训练输出：

- `checkpoints/<run_name>.pt`：最佳模型 checkpoint
- `checkpoints/<run_name>_history.json`：训练与验证历史

推理输出：

- `test.py` 默认会将预测结果写到 `<DATASET_DIR>/<DATASET>.txt`

## 说明

- 数据集文件不包含在仓库中，需要自行准备。
