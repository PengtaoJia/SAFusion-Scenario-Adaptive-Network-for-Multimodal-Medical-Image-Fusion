# SAFusion: 多模态医学图像融合框架

## 简介

SAFusion 是一个用于多模态医学图像融合的深度学习框架。该框架能够处理多种医学图像对，包括 MRI-CT、MRI-PET 和 MRI-SPECT 图像对，生成高质量的融合图像，以辅助医生诊断和治疗规划。

## 预训练权重

链接: https://pan.baidu.com/s/199CmQNiNS3etcIrWg-PGKg?pwd=uiss 提取码: uiss

## 目录结构

```
# SAFusion: 多模态医学图像融合框架
## 使用方法

cd FusionNet
python test.py
```

测试脚本会自动加载预训练的 AutoEncoder 和 Fusion 模型，并对 [images/](file:///C:/Users/11352/Desktop/SAFusion/images/) 目录下的示例图像进行融合，结果保存在 [fusion_result/](file:///C:/Users/11352/Desktop/SAFusion/fusion_result/) 目录中，按照不同的图像对类型分别存储：

- [fusion_result/MRI-CT/](file:///C:/Users/11352/Desktop/SAFusion/fusion_result/MRI-CT/)
- [fusion_result/MRI-PET/](file:///C:/Users/11352/Desktop/SAFusion/fusion_result/MRI-PET/)
- [fusion_result/MRI-SPECT/](file:///C:/Users/11352/Desktop/SAFusion/fusion_result/MRI-SPECT/)
# SAFusion-Scenario-Adaptive-Network-for-Multimodal-Medical-Image-Fusion
