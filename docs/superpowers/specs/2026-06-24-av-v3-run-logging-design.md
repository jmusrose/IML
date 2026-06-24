# AV_v3 独立运行日志设计

## 目标

CREMA-D、KS 和 AVE 每次启动训练时创建独立运行目录，完整保存参数、逐轮指标、控制台日志、曲线、检查点和最终指标，不覆盖历史实验。

## 目录结构

`--output-dir` 表示实验根目录。训练启动后创建时间戳子目录：

```text
runs/ave_baseline/
  20260624_091530/
    config.json
    train.log
    history.jsonl
    history.json
    curves.png
    train_loss_curves.png
    val_loss_curves.png
    train_modality_accuracy.png
    val_modality_accuracy.png
    best.pt
    metrics.json
```

若同一秒内目录已存在，依次使用 `_01`、`_02` 后缀。

## 行为

- 三个训练入口调用共享函数创建运行目录。
- `config.json` 记录原始参数，并额外记录实际运行目录和启动时间。
- 训练期间的标准输出与标准错误同时显示在终端并追加写入 `train.log`。
- 训练异常时仍保留已写入的参数、日志、历史和曲线。
- 用户显式传入的 `--output-dir` 仍是根目录，不直接作为单次运行目录。
- 旧目录和旧实验文件不迁移、不删除、不覆盖。

## 测试

- 连续创建两次运行目录时路径不同。
- 时间戳冲突时生成带编号后缀的目录。
- 三个训练入口把所有产物写入生成的运行目录。
- `train.log` 同时记录普通输出和异常信息。
