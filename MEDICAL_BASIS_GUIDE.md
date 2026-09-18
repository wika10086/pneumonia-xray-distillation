# 医学参考依据说明

项目现在会在预测结果中加入“医学参考依据”。这部分不是自动诊断，而是帮助你知道医生通常会怎样复核这个结果。

## 肺炎判断通常需要综合信息

肺炎诊断一般不能只依靠模型概率，需要结合：

- 临床症状：发热、咳嗽、咳痰、胸痛、气促等。
- 体征：听诊湿啰音、呼吸音改变、血氧下降等。
- 实验室检查：血常规、CRP、降钙素原、病原学检查等。
- 影像学证据：胸片或 CT 上的新发浸润影、实变影、空气支气管征等。

## 输出中的医学复核内容

如果模型预测为 `pneumonia`，程序会提示医生或人工复核：

- 模型关注区域是否有浸润影、实变影。
- 是否有空气支气管征。
- 是否有胸腔积液或胸膜反应。
- 影像表现是否和发热、咳嗽、咳痰、气促等临床表现一致。

如果模型预测为 `normal`，程序也会提示：

- 如果症状明显，不能仅凭模型结果排除肺炎。
- 早期、轻微、隐匿或被遮挡的病变可能不明显。
- 必要时应由医生决定是否复查胸片、CT 或进一步检查。

## 重要限制

- 这些内容是学习和复核清单，不是诊断结论。
- Grad-CAM 关注区域不等于真实病灶位置。
- 真实医学诊断必须由医生结合完整病史、查体、检验和影像报告完成。

## 参考资料

- ATS/IDSA Community-Acquired Pneumonia Guideline: https://www.atsjournals.org/doi/10.1164/rccm.201908-1581ST
- CDC Pneumonia: https://www.cdc.gov/pneumonia/
- Merck Manual Professional, Overview of Pneumonia: https://www.merckmanuals.com/professional/pulmonary-disorders/pneumonia/overview-of-pneumonia
