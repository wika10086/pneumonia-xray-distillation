# 数据集与去重说明

本项目当前基础数据是 `D:\picdata` 中的 PneumoniaMNIST 224x224。为了增加训练图片数量，新增脚本会下载一个独立来源的胸片数据集，然后先去重再导入训练集。

## 新增数据集

- 名称：A Primary Chest X-ray Dataset of Normal and Pneumonia Cases from Epic Chittagong, Bangladesh
- 来源：[Mendeley Data](https://data.mendeley.com/datasets/wndbd5r26y)
- DOI：`10.17632/wndbd5r26y.4`
- 规模：3355 张胸片，包含 `Normal` 和 `Pneumonia`
- 许可证：CC BY 4.0

## 为什么不直接继续下载同类 PneumoniaMNIST 数据

PneumoniaMNIST 来自常见的 Kermany 儿童胸片数据源。很多网上的“Chest X-Ray Pneumonia”压缩包都和它高度同源，直接加入容易把重复图片混进训练集，导致模型看起来变好但实际泛化能力没有提高。

## 下载和导入

```powershell
.\.venv\Scripts\python.exe download_epic_chittagong.py
```

脚本会做这些事：

1. 下载压缩包到 `D:\picdata\raw\epic_chittagong`。
2. 如果压缩包已存在，并且 SHA256 校验通过，就不会重复下载。
3. 解压到 `D:\picdata\raw\epic_chittagong\extracted`。
4. 扫描现有 `train`、`val`、`test` 图片，建立去重索引。
5. 对新增图片做完全哈希和感知哈希检查，默认近似重复阈值为 `2`。
6. 只把非重复图片复制进 `D:\picdata\train\normal` 和 `D:\picdata\train\pneumonia`。

正式导入报告会保存到：

```text
outputs\epic_chittagong_import_report.json
outputs\epic_chittagong_import_rows.csv
```

## 只试运行，不真正复制图片

```powershell
.\.venv\Scripts\python.exe download_epic_chittagong.py --dry-run
```

试运行报告会保存到：

```text
outputs\epic_chittagong_dry_run_report.json
outputs\epic_chittagong_dry_run_rows.csv
```

## 训练时启用数据增强

```powershell
.\.venv\Scripts\python.exe train_student.py --epochs 8 --device cuda --pretrained --augment
```

后续蒸馏训练也可以加上 `--augment`：

```powershell
.\.venv\Scripts\python.exe distill.py --epochs 8 --device cuda --student-pretrained --alpha 0.3 --temperature 2 --checkpoint-name distilled_student_augmented_best.pt --history-name distill_augmented_history.json --augment
```
