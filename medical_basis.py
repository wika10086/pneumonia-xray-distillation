from __future__ import annotations

from typing import Any


SOURCE_REFERENCES = [
    {
        "name": "ATS/IDSA community-acquired pneumonia guideline",
        "url": "https://www.atsjournals.org/doi/10.1164/rccm.201908-1581ST",
        "note": "社区获得性肺炎诊疗指南，强调临床表现与影像证据的综合判断。",
    },
    {
        "name": "CDC pneumonia overview",
        "url": "https://www.cdc.gov/pneumonia/",
        "note": "肺炎症状、风险和就医建议的公众健康资料。",
    },
    {
        "name": "Merck Manual pneumonia diagnosis",
        "url": "https://www.merckmanuals.com/professional/pulmonary-disorders/pneumonia/overview-of-pneumonia",
        "note": "肺炎诊断通常结合症状、体征和胸部影像。",
    },
]


COMMON_IMAGE_FINDINGS = [
    "肺野内新发或局灶性浸润影、斑片状阴影",
    "肺实变影，也就是肺泡内被炎性渗出物填充后形成的较白区域",
    "空气支气管征，即白色实变区域中仍能看到较暗的支气管气影",
    "间质性或网状阴影增多，部分病毒性或非典型感染可见",
    "胸腔积液或胸膜反应，但这不是所有肺炎都会出现",
]


CLINICAL_ITEMS_TO_CONFIRM = [
    "是否有发热、寒战、咳嗽、咳痰、胸痛、气促等症状",
    "听诊是否有湿啰音、呼吸音改变等体征",
    "血常规、CRP、降钙素原、病原学检测等是否支持感染",
    "是否有基础疾病、免疫抑制、近期感染接触史或住院史",
    "必要时由医生决定是否需要复查胸片、CT 或进一步检查",
]


DIFFERENTIAL_CONSIDERATIONS = [
    "肺不张",
    "肺水肿",
    "肺结核",
    "肺部肿瘤或占位",
    "慢性炎症或陈旧病灶",
    "拍摄体位、曝光不足、旋转、遮挡物等技术因素",
]


def build_medical_basis(
    *,
    prediction: str,
    positive_probability: float,
    threshold: float,
    focus_regions: list[str] | None = None,
) -> dict[str, Any]:
    focus_text = "、".join(focus_regions or []) if focus_regions else "模型未生成明确关注区域"

    if prediction == "pneumonia":
        summary = (
            "医学复核方向：模型提示肺炎概率达到判断阈值，但医学诊断仍需要医生结合症状、体征、"
            "实验室检查和胸部影像征象综合判断。"
        )
        image_review = [
            f"优先复核模型关注区域：{focus_text}。",
            "查看这些区域是否存在浸润影、实变影、空气支气管征或胸腔积液等肺炎相关征象。",
            "如果影像异常与发热、咳嗽、咳痰、气促等临床表现一致，肺炎可能性会更高。",
        ]
    else:
        summary = (
            "医学复核方向：模型未达到肺炎判断阈值，但如果临床症状明显或风险较高，"
            "仍不能只凭模型结果排除肺炎。"
        )
        image_review = [
            f"可复核模型关注区域：{focus_text}。",
            "查看胸片是否仍有轻微、早期、隐匿或被遮挡的浸润影。",
            "如果患者有明显发热、咳嗽、气促、低氧等表现，应由医生决定是否复查影像或进一步检查。",
        ]

    return {
        "summary": summary,
        "model_known_facts": [
            f"模型输出类别：{prediction}",
            f"pneumonia 概率：{positive_probability:.4f}",
            f"当前判断阈值：{threshold:.4f}",
        ],
        "image_findings_to_review": image_review + COMMON_IMAGE_FINDINGS,
        "clinical_items_to_confirm": CLINICAL_ITEMS_TO_CONFIRM,
        "differential_considerations": DIFFERENTIAL_CONSIDERATIONS,
        "references": SOURCE_REFERENCES,
        "warning": "本项目只能提供学习用途的医学复核清单，不能替代医生诊断、影像科报告或治疗建议。",
    }


def medical_basis_text(medical_basis: dict[str, Any], max_items: int = 4) -> str:
    lines = [medical_basis["summary"]]
    findings = medical_basis.get("image_findings_to_review", [])[:max_items]
    if findings:
        lines.append("建议复核：" + "；".join(findings))
    lines.append(medical_basis["warning"])
    return "\n".join(lines)
