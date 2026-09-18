# 图像识别蒸馏项目

这是一个使用 PyTorch 的基础图像识别与知识蒸馏项目。当前环境已经配置好，可以使用 NVIDIA GPU 训练模型。该项目只用于学习和实验，**不能用于真实临床诊断**。


当前项目已准备好 **PneumoniaMNIST 224x224** 数据集，并在训练集中加入了经过去重的 Epic Chittagong 胸片。
该仓库只包含源码和说明文档。数据集、模型权重、训练输出、虚拟环境和打包好的 exe 均需在本机另行准备；单独克隆源码不能直接运行预测界面。

## 当前环境

- Python: 3.13.14
- PyTorch: 2.11.0+cu128
- torchvision: 0.26.0+cu128
- torchaudio: 2.11.0+cu128
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- 默认数据目录: `D:\picdata`
- 当前数据集: PneumoniaMNIST 224x224 + 去重后的 Epic Chittagong 训练图片

## 在其他电脑准备环境

以下命令以 Windows PowerShell 为例。先安装 Python 3.13，再在项目目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

脚本默认使用 `D:\picdata`。请先准备该目录中的数据；数据来源与导入步骤见 [DATASET_GUIDE.md](DATASET_GUIDE.md)。未训练并生成模型权重前，预测脚本和图形界面无法使用。

## 数据集结构

```text
D:\picdata
  train
    normal
    pneumonia
  val
    normal
    pneumonia
  test
    normal
    pneumonia
  raw
    pneumoniamnist.npz
    pneumoniamnist_224.npz
```

PneumoniaMNIST 是胸部 X 光二分类数据集。当前类别为：

- `normal`: 正常胸片
- `pneumonia`: 肺炎胸片

图片数量：

- `train`: 6600 张（包含去重后导入的 1892 张 Epic Chittagong 胸片）
- `val`: 524 张
- `test`: 624 张

## 重新准备数据

默认重新生成 224x224 版本。此操作会清除 `train`、`val`、`test` 类别目录中已有的 PNG 图片；如需保留扩充后的训练集，请先备份并在重建后重新导入：

```powershell
.\.venv\Scripts\python.exe prepare_pneumoniamnist.py
```

如果要改成其他尺寸：

```powershell
.\.venv\Scripts\python.exe prepare_pneumoniamnist.py --size 128
```

可选尺寸包括 `28`、`64`、`128`、`224`。

## 复查环境

```powershell
.\.venv\Scripts\python.exe check_environment.py
```

如果看到 `CUDA available: True` 和 `Device tensor test: ... on cuda:0`，说明 PyTorch 可以使用显卡。

## 训练教师模型

教师模型使用 ResNet18；以下命令对应当前增强版模型：

```powershell
.\.venv\Scripts\python.exe train_teacher.py --epochs 6 --device cuda --pretrained --augment --checkpoint-name teacher_augmented_best.pt --history-name teacher_augmented_history.json
```

最佳教师模型会保存到：

```text
checkpoints\teacher_augmented_best.pt
```

## 训练普通学生模型

学生模型默认使用 MobileNetV3-Small：

```powershell
.\.venv\Scripts\python.exe train_student.py --epochs 5 --device cuda --pretrained
```

最佳普通学生模型会保存到：

```text
checkpoints\student_best.pt
```

## 蒸馏训练学生模型

蒸馏前需要先训练并保存教师模型。以下命令对应当前增强版蒸馏模型：

```powershell
.\.venv\Scripts\python.exe distill.py --epochs 8 --device cuda --teacher-checkpoint checkpoints/teacher_augmented_best.pt --student-pretrained --augment --alpha 0.3 --temperature 2 --checkpoint-name distilled_student_augmented_best.pt --history-name distill_augmented_history.json
```

最佳蒸馏学生模型会保存到：

```text
checkpoints\distilled_student_augmented_best.pt
```

## 测试集评估

训练和蒸馏完成后，使用 `test` 数据集评估模型：

```powershell
.\.venv\Scripts\python.exe evaluate_models.py --device cuda --checkpoints checkpoints/teacher_augmented_best.pt checkpoints/distilled_student_augmented_best.pt --output-name test_metrics_augmented.json
```

测试结果会保存到：

```text
outputs\test_metrics_augmented.json
```

当前测试集结果：

| 模型 | Test accuracy | Macro F1 | normal recall | pneumonia recall |
| --- | ---: | ---: | ---: | ---: |
| 教师 ResNet18 | 0.9375 | 0.9309 | 0.8376 | 0.9974 |
| 普通学生 MobileNetV3-Small | 0.9103 | 0.8985 | 0.7607 | 1.0000 |
| 第一次蒸馏 alpha=0.7, temperature=4 | 0.9087 | 0.8969 | 0.7607 | 0.9974 |
| 优化蒸馏 alpha=0.3, temperature=2 | 0.9151 | 0.9043 | 0.7735 | 1.0000 |
| 增强版教师 ResNet18 | 0.9407 | 0.9346 | 0.8462 | 0.9974 |
| 增强版蒸馏学生 MobileNetV3-Small | 0.9439 | 0.9385 | 0.8632 | 0.9923 |

验证集准确率和测试集准确率不同是正常的。测试集没有参与训练和调参，更适合用来判断模型的最终泛化能力。

## 阈值优化

二分类模型默认使用 `0.5` 作为判断阈值：如果 pneumonia 概率大于等于 `0.5`，就预测为肺炎。

可以在验证集上自动选择阈值，再应用到测试集。当前图形界面的默认值来自增强版模型、验证集最低 sensitivity 0.98 的一次分析：

```powershell
.\.venv\Scripts\python.exe threshold_analysis.py --device cuda --checkpoint checkpoints/distilled_student_augmented_best.pt --min-sensitivity 0.98 --output-name threshold_metrics_augmented_sens098.json
```

脚本的参数默认目标是保持 pneumonia 的 sensitivity 不低于 `0.99`；上面的命令明确指定 `0.98`，与当前界面使用的阈值一致。

增强版蒸馏模型的阈值结果：

| 设置 | 阈值 | Test accuracy | ROC-AUC | Sensitivity | Specificity | 混淆矩阵 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 默认阈值 | 0.500000 | 0.9439 | 0.9888 | 0.9923 | 0.8632 | `[[202, 32], [3, 387]]` |
| 当前界面阈值，验证集目标 0.98 | 0.520043 | 0.9439 | 0.9888 | 0.9923 | 0.8632 | `[[202, 32], [3, 387]]` |

当前演示界面使用 `0.520043`。这是在验证集上选出的阈值；测试集仍有 3 张肺炎胸片被漏判，不能把它视作临床安全阈值。

## 单张图片预测

使用优化蒸馏模型和推荐阈值预测一张图片：

```powershell
.\.venv\Scripts\python.exe predict_image.py --image D:\picdata\test\pneumonia\test_00000.png --device cuda
```

默认设置：

- 模型: `checkpoints\distilled_student_augmented_best.pt`
- 阈值: `0.520043`
- 阳性类别: `pneumonia`
- 输出文件: `outputs\predictions.json`

也可以指定输出文件：

```powershell
.\.venv\Scripts\python.exe predict_image.py --image D:\picdata\test\normal\test_00003.png --device cuda --output outputs\prediction_normal_correct_sample.json
```

脚本会输出：

- 预测类别: `normal` 或 `pneumonia`
- pneumonia 概率
- 当前使用的阈值
- 每个类别的概率

也可以输入一个文件夹进行批量预测：

```powershell
.\.venv\Scripts\python.exe predict_image.py --image D:\picdata\test --device cuda --output outputs\test_predictions.json
```

## 图形界面和 exe

源码方式启动图形界面：

```powershell
.\.venv\Scripts\python.exe app_gui.py
```

已经打包好的 exe 在：

```text
dist\PneumoniaPredictor\PneumoniaPredictor.exe
```

注意：不要只复制单独的 `PneumoniaPredictor.exe` 文件。因为 PyTorch 依赖很大，必须复制整个文件夹：

```text
dist\PneumoniaPredictor
```

图形界面功能：

- 选择单张图片进行预测。
- 选择文件夹进行批量预测。
- 显示图片预览。
- 显示 `normal` / `pneumonia` 预测结果。
- 显示 pneumonia 概率和当前阈值。
- 保存预测结果为 JSON。

重新打包 exe：

```powershell
.\build_exe.ps1
```

当前打包目录约 4.5 GB，主要原因是打包了 GPU 版 PyTorch 和 CUDA 依赖。

### exe 报错排查

如果 Windows 提示 `PneumoniaPredictor.exe` 崩溃，并且事件查看器里显示崩溃模块是 `msvcp140.dll`，原因通常是 Visual C++ 运行库版本冲突。

本项目已经在 `PneumoniaPredictor.spec` 中排除了 PyInstaller 捆绑的旧版：

- `msvcp140.dll`
- `vcruntime140.dll`
- `vcruntime140_1.dll`

当前电脑会改用系统里的新版 Visual C++ 运行库。已验证新版 exe 能正常启动并关闭。

如果把软件复制到其他电脑后仍然打不开，请先安装 Microsoft Visual C++ 2015-2022 Redistributable x64。

如果界面里出现模型加载或预测错误，程序会在 exe 同级目录生成：

```text
PneumoniaPredictor_error.log
```

可以查看这个文件定位具体原因。

## 文件说明

- `prepare_pneumoniamnist.py`: 下载并转换 PneumoniaMNIST。
- `config.py`: 公共训练参数。
- `dataset.py`: 读取 `D:\picdata` 中的图片数据。
- `models.py`: 创建教师模型和学生模型。
- `engine.py`: 普通训练、验证、蒸馏训练的核心循环。
- `train_teacher.py`: 训练教师模型。
- `train_student.py`: 训练普通学生模型。
- `distill.py`: 用教师模型蒸馏训练学生模型。
- `evaluate_models.py`: 在测试集上评估多个模型。
- `threshold_analysis.py`: 在验证集上优化二分类阈值，并在测试集上评估。
- `predict_image.py`: 对单张图片或文件夹中的图片进行预测。
- `app_gui.py`: 简单图形界面。
- `PneumoniaPredictor.spec`: PyInstaller 打包配置。
- `build_exe.ps1`: 一键打包脚本。
- `utils.py`: 保存模型、保存日志、设置随机种子等工具函数。
- `check_environment.py`: 检查 PyTorch 和 CUDA 是否可用。
