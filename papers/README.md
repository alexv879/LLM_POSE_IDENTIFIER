# Downloaded Research Papers

✅ **All 18 papers are included in this repository (~200MB total)**

This directory contains all the research papers referenced in the Pose LLM Identifier project. Unlike most repositories that only link to papers, we include the full PDFs for offline access and convenience.

## � What's Included

All PDFs are in this directory. You don't need to download anything - they're ready to read!

**Total size: ~200MB (18 PDFs)**

## 📥 Papers Already Included

## 📥 Papers Already Included

No download needed - all papers are in this directory!

### Option 1: Browse Locally ✅ (Recommended)

All PDFs are ready to read:
```
papers/
├── Sapiens_2B_ECCV2024.pdf          (17.2 MB) ✅
├── ViTPose_NeurIPS2022.pdf          (1.94 MB) ✅
├── DWPose_ICCV2023.pdf              (5.95 MB) ✅
├── SSL_MultiPath_ICLR2025.pdf       (2.15 MB) ✅
├── HRNet_CVPR2019.pdf               ✅
├── OpenPose_CVPR2017.pdf            ✅
├── ViT_ICLR2021.pdf                 ✅
├── MAE_CVPR2022.pdf                 ✅
├── COCO_Dataset_ECCV2014.pdf        ✅
├── ... (9 more papers)              ✅
```

### Option 2: Online Links (If Needed)

If you prefer to read online or need updated versions, see links below.

---

## 🔥 Priority 1: Essential Papers (READ THESE FIRST!)

### 1. Sapiens-2B (ECCV 2024 Oral) ⭐⭐⭐
- **Why Essential**: This is THE foundation model we're using
- **File**: `Sapiens_2B_ECCV2024.pdf`
- **URL**: https://arxiv.org/pdf/2408.12569.pdf
- **What to Read**: 
  - Section 3: Architecture (ViT-2B + MAE pretraining)
  - Section 4.2: Pose estimation methodology
  - Section 5: Results on COCO (82-84% zero-shot!)
  - Table 3: Performance comparison

### 2. ViTPose (NeurIPS 2022) ⭐⭐⭐
- **Why Essential**: Ensemble component, pure transformer baseline
- **File**: `ViTPose_NeurIPS2022.pdf`
- **URL**: https://arxiv.org/pdf/2204.12004.pdf
- **What to Read**:
  - Section 3.1: Vision Transformer for pose
  - Section 3.3: Training strategy
  - Table 2: COCO results (81-82% AP)
  - Figure 3: Architecture diagram

### 3. DWPose (ICCV 2023) ⭐⭐⭐
- **Why Essential**: Ensemble component, knowledge distillation
- **File**: `DWPose_ICCV2023.pdf`
- **URL**: https://arxiv.org/pdf/2307.11573.pdf
- **What to Read**:
  - Section 3.2: Knowledge distillation process
  - Section 4: Whole-body pose (133 keypoints)
  - Table 1: Performance on COCO-WholeBody
  - Equation 3: Distillation loss formula

### 4. SSL Multi-Path (ICLR 2025) ⭐⭐⭐
- **Why Essential**: Our Stage 2 methodology for SSL
- **File**: `SSL_MultiPath_ICLR2025.pdf`
- **URL**: https://openreview.net/pdf?id=5zGuFj0y9V
- **What to Read**:
  - Section 3.2: Multi-path consistency regularization
  - Section 3.3: Augmentation synergies (+5-7% improvement!)
  - Algorithm 1: Training procedure
  - Table 2: Ablation studies

---

## 📚 Priority 2: Important Supporting Papers

### 5. HRNet (CVPR 2019)
- **Purpose**: High-resolution CNN baseline
- **File**: `HRNet_CVPR2019.pdf`
- **URL**: https://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Deep_High-Resolution_Representation_Learning_for_Visual_Recognition_CVPR_2019_paper.pdf
- **Key Sections**: Section 3 (parallel multi-resolution architecture)

### 6. OpenPose (CVPR 2017)
- **Purpose**: Classic multi-person pose, Part Affinity Fields
- **File**: `OpenPose_CVPR2017.pdf`
- **URL**: https://openaccess.thecvf.com/content_cvpr_2017/papers/Cao_Realtime_Multi-Person_2D_CVPR_2017_paper.pdf
- **Key Sections**: Section 3.1 (PAF definition), Figure 2 (architecture)

### 7. Vision Transformer (ICLR 2021)
- **Purpose**: Foundation for ViTPose and Sapiens
- **File**: `ViT_ICLR2021.pdf`
- **URL**: https://arxiv.org/pdf/2010.11929.pdf
- **Key Sections**: Section 3 (Transformer architecture), Figure 1

### 8. MAE (CVPR 2022)
- **Purpose**: Masked autoencoder pretraining (used in Sapiens)
- **File**: `MAE_CVPR2022.pdf`
- **URL**: https://arxiv.org/pdf/2111.06377.pdf
- **Key Sections**: Section 3 (MAE methodology), Figure 1 (masking)

### 9. COCO Dataset (ECCV 2014)
- **Purpose**: Dataset specification, OKS metric definition
- **File**: `COCO_Dataset_ECCV2014.pdf`
- **URL**: https://arxiv.org/pdf/1405.0312.pdf
- **Key Sections**: Section 5 (keypoint format), Equation 3 (OKS)

### 10. SDPose (Pre-print)
- **Purpose**: Diffusion-based pose (alternative to Sapiens)
- **File**: `SDPose_Diffusion.pdf`
- **URL**: https://arxiv.org/pdf/2509.24980.pdf
- **Key Sections**: Section 3.2 (Stable Diffusion adaptation)

### 11. UniPose (Pre-print)
- **Purpose**: Multimodal LLM for pose
- **File**: `UniPose_Multimodal.pdf`
- **URL**: https://arxiv.org/pdf/2411.16781.pdf
- **Key Sections**: Section 3.3 (pose tokenization with VQ-VAE)

---

## 📖 Priority 3: Background & Historical Context

### 12. DeepPose (CVPR 2014)
- **Purpose**: First CNN-based pose estimation
- **File**: `DeepPose_CVPR2014.pdf`
- **URL**: https://arxiv.org/pdf/1312.4659.pdf

### 13. Stacked Hourglass (ECCV 2016)
- **Purpose**: Recursive refinement architecture
- **File**: `HourglassNetworks_ECCV2016.pdf`
- **URL**: https://arxiv.org/pdf/1603.06937.pdf

### 14. CPM - Convolutional Pose Machines (CVPR 2016)
- **Purpose**: Sequential confidence maps
- **File**: `CPM_CVPR2016.pdf`
- **URL**: https://arxiv.org/pdf/1602.00134.pdf

### 15. Simple Baseline (ECCV 2018)
- **Purpose**: Simple > Complex philosophy
- **File**: `SimpleBaseline_ECCV2018.pdf`
- **URL**: https://arxiv.org/pdf/1804.06208.pdf

### 16. Knowledge Distillation (NIPS 2014)
- **Purpose**: Hinton's original distillation paper
- **File**: `KnowledgeDistillation_Hinton.pdf`
- **URL**: https://arxiv.org/pdf/1503.02531.pdf

### 17. Pose Survey (2022)
- **Purpose**: Comprehensive survey of all methods
- **File**: `Pose_Survey_2022.pdf`
- **URL**: https://arxiv.org/pdf/2204.07370.pdf

---

## 📋 Reading Order (Recommended)

### For Implementation:
1. **Sapiens-2B** - Understand the foundation model
2. **SSL Multi-Path** - Understand Stage 2 methodology
3. **ViTPose** - Understand ensemble component 1
4. **DWPose** - Understand ensemble component 2
5. **COCO Dataset** - Understand data format and metrics

### For Research/Thesis:
1. **Pose Survey** - Get historical context
2. **DeepPose** - Understand evolution from 2014
3. **HRNet** - Understand CNN baselines
4. **ViT + MAE** - Understand transformer foundations
5. **Sapiens-2B** - Understand current SOTA
6. **SSL Multi-Path** - Understand our contribution

### For Deep Dive:
- Read ALL papers in priority order
- Compare Tables 1-3 in each paper
- Understand equations and loss functions
- Analyze failure cases in supplementary materials

---

## 📊 Paper Statistics

| Category | Count | Total Size |
|----------|-------|------------|
| Priority 1 (Essential) | 4 papers | ~20 MB |
| Priority 2 (Important) | 7 papers | ~30 MB |
| Priority 3 (Background) | 6 papers | ~50 MB |
| **TOTAL** | **17 papers** | **~100 MB** |

---

## 🔍 Quick Reference Table

| Paper | Year | Venue | Method | AP (COCO) | Parameters |
|-------|------|-------|--------|-----------|------------|
| DeepPose | 2014 | CVPR | CNN Regression | ~60% | 7M |
| CPM | 2016 | CVPR | Sequential Heatmaps | ~70% | 100M |
| Hourglass | 2016 | ECCV | Recursive | ~75% | 25M |
| OpenPose | 2017 | CVPR | PAF | ~60% | 200M |
| SimpleBaseline | 2018 | ECCV | ResNet+Deconv | ~73% | 34M |
| HRNet | 2019 | CVPR | High-Res Parallel | ~76% | 28M |
| ViT | 2021 | ICLR | Pure Transformer | - | 86M |
| ViTPose | 2022 | NeurIPS | ViT for Pose | ~81% | 100M |
| DWPose | 2023 | ICCV | Distillation | ~66% (whole-body) | 50M |
| **Sapiens-2B** | **2024** | **ECCV** | **Foundation Model** | **82-92%** | **2000M** |
| SSL Multi-Path | 2025 | ICLR | SSL Augmentation | +5-7% boost | - |

---

## 🛠️ Tools & Code Repositories

### Implementation Code:
- **Sapiens**: https://github.com/facebookresearch/sapiens
- **ViTPose**: https://github.com/ViTAE-Transformer/ViTPose
- **DWPose**: https://github.com/IDEA-Research/DWPose
- **HRNet**: https://github.com/HRNet/HRNet-Image-Classification
- **OpenPose**: https://github.com/CMU-Perceptual-Computing-Lab/openpose
- **MMPose** (Unified): https://github.com/open-mmlab/mmpose

### Annotation Tools:
- **CVAT**: https://github.com/opencv/cvat
- **LabelMe**: https://github.com/wkentaro/labelme

### Utilities:
- **COCO API**: https://github.com/cocodataset/cocoapi
- **Albumentations**: https://github.com/albumentations-team/albumentations

---

## 💡 Citation Format

When citing these papers in your thesis:

```bibtex
@inproceedings{sapiens2024,
  title={Sapiens: Foundation for Human Vision Models},
  author={Khirodkar, Rawal and Bagautdinov, Timur and Martinez, Julieta and others},
  booktitle={ECCV},
  year={2024}
}

@inproceedings{vitpose2022,
  title={ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation},
  author={Xu, Yufei and Zhang, Jing and Zhang, Qiming and Tao, Dacheng},
  booktitle={NeurIPS},
  year={2022}
}

@inproceedings{dwpose2023,
  title={Effective Whole-body Pose Estimation with Two-stages Distillation},
  author={Yang, Zhendong and Zeng, Ailing and Yuan, Chun and Li, Yu},
  booktitle={ICCV},
  year={2023}
}
```

---

## 🚀 Next Steps

1. **Download papers**: Run the download script
2. **Read Priority 1 papers**: Focus on Sapiens, SSL, ViTPose, DWPose
3. **Understand methodology**: Extract key equations and architectures
4. **Compare with implementation**: Match paper descriptions to code
5. **Write thesis sections**: Use papers as references

---

## ❓ FAQ

**Q: Which papers should I read first?**
A: Start with Priority 1 (4 papers, ~2-3 hours). These are essential for implementation.

**Q: Do I need to read all 17 papers?**
A: For implementation: No (Priority 1 sufficient). For thesis: Yes (background important).

**Q: Papers won't download?**
A: Some conference PDFs may be behind paywalls. Use arXiv versions or access via university.

**Q: Can I use these papers for my thesis?**
A: Yes! They're all published and publicly available. Just cite properly.

**Q: What if download script fails?**
A: Download manually from URLs above. Right-click → Save link as...

---

**Last Updated**: November 5, 2025
**Total Papers**: 17
**Status**: ✅ All URLs verified and working
