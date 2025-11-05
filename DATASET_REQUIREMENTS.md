# 📦 Dataset Requirements - MUST DOWNLOAD

⚠️ **IMPORTANT:** Datasets are NOT included in this repository due to size limitations. You must download them separately before training.

---

## 🔴 REQUIRED: COCO 2017 Dataset (~19GB)

### **What You Need:**
- **train2017**: 118,287 images (~18GB)
- **val2017**: 5,000 images (~1GB)
- **Annotations**: Person keypoints JSON files (~241MB)

### **Download Options:**

#### **Option A: Automatic (Recommended)**
```bash
cd LLM_POSE_IDENTIFIER
python scripts/download_coco.py
```

This will:
✅ Download train2017 images  
✅ Download val2017 images  
✅ Download keypoint annotations  
✅ Extract to correct folders  
✅ Verify file integrity  

**Time:** 30-60 minutes (depending on internet speed)

#### **Option B: Manual Download**

1. **Visit:** https://cocodataset.org/#download

2. **Download these files:**
   - 2017 Train images [18GB]: http://images.cocodataset.org/zips/train2017.zip
   - 2017 Val images [1GB]: http://images.cocodataset.org/zips/val2017.zip
   - 2017 Train/Val annotations [241MB]: http://images.cocodataset.org/annotations/annotations_trainval2017.zip

3. **Extract to this structure:**
   ```
   LLM_POSE_IDENTIFIER/
   └── data/
       └── coco/
           ├── images/
           │   ├── train2017/        (118,287 .jpg files)
           │   └── val2017/          (5,000 .jpg files)
           └── annotations/
               ├── person_keypoints_train2017.json
               └── person_keypoints_val2017.json
   ```

### **Verify Installation:**
```bash
python scripts/validate_setup.py
```

Should show:
```
✓ COCO train2017: 118,287 images found
✓ COCO val2017: 5,000 images found
✓ Annotations: person_keypoints_train2017.json (valid JSON)
✓ Annotations: person_keypoints_val2017.json (valid JSON)
```

---

## 🟡 OPTIONAL: Unlabeled COCO for SSL (~5GB)

**Only needed for Stage 2 Semi-Supervised Learning**

### **What It's For:**
Stage 2 uses unlabeled images with multi-path augmentation for consistency regularization. This can improve performance by +5-7% AP.

### **Download:**
```bash
python scripts/download_coco_unlabeled.py --num_images 5000
```

This downloads 5,000 additional COCO images (without annotations) to:
```
data/external/coco_unlabeled/
```

**Time:** 10-15 minutes

**Skip if:** You only want to run Stage 1 baseline training

---

## 🟢 AUTO-DOWNLOADED: Pretrained Sapiens Weights (~5GB)

**No action needed!** Weights automatically download on first run.

### **Details:**
- **Model:** facebook/sapiens-pretrain-1b-torchscript
- **Size:** ~5GB
- **Storage:** `~/.cache/huggingface/hub/`
- **When:** First time you run training

### **To Pre-download (Optional):**
```bash
python scripts/download_pretrained.py
```

---

## 📊 Summary Table

| Dataset | Size | Required? | Download Command | Used In |
|---------|------|-----------|------------------|---------|
| **COCO train2017** | ~18GB | ✅ YES | `python scripts/download_coco.py` | All stages |
| **COCO val2017** | ~1GB | ✅ YES | (included in above) | Validation |
| **COCO annotations** | ~241MB | ✅ YES | (included in above) | All stages |
| **Unlabeled COCO** | ~5GB | ⚠️ OPTIONAL | `python scripts/download_coco_unlabeled.py` | Stage 2 SSL only |
| **Sapiens-1B weights** | ~5GB | ✅ YES | Auto-downloads | All stages |
| **Research papers** | ~200MB | ✅ INCLUDED | Already in repo | Reference |

---

## 💾 Disk Space Requirements

### **Minimum (Stage 1 only):**
- Repository: ~300MB (code + papers)
- COCO dataset: ~19GB
- Pretrained weights: ~5GB
- Training outputs: ~1GB (checkpoints, logs)
- **Total: ~25GB**

### **Full Setup (All stages):**
- Repository: ~300MB
- COCO dataset: ~19GB
- Unlabeled COCO: ~5GB
- Pretrained weights: ~5GB
- Training outputs: ~2GB
- **Total: ~31GB**

---

## 🚨 Common Issues

### **"COCO dataset not found" error**
**Solution:** Run `python scripts/download_coco.py` before training

### **"No space left on device"**
**Solution:** Free up at least 30GB of disk space

### **Download is very slow**
**Solutions:**
1. Use manual download with a download manager
2. Try during off-peak hours
3. Use university/institutional connection if available

### **Annotations file corrupted**
**Solution:** 
```bash
# Re-download just annotations
cd data/coco
rm -rf annotations/
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### **Wrong directory structure**
**Solution:** Follow the exact structure shown above. Use `validate_setup.py` to check.

---

## 📖 COCO Dataset Format

### **Keypoint Format:**
```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "keypoints": [x1,y1,v1, x2,y2,v2, ..., x17,y17,v17],
      "bbox": [x, y, width, height],
      "area": 12345,
      "iscrowd": 0
    }
  ],
  "categories": [...]
}
```

### **17 COCO Keypoints:**
```
0: nose          5: left_shoulder   11: left_hip      15: left_ankle
1: left_eye      6: right_shoulder  12: right_hip     16: right_ankle
2: right_eye     7: left_elbow      13: left_knee
3: left_ear      8: right_elbow     14: right_knee
4: right_ear     9: left_wrist
                10: right_wrist
```

### **Visibility Values:**
- `v=0`: Not labeled (keypoint not in image)
- `v=1`: Labeled but not visible (occluded)
- `v=2`: Labeled and visible

---

## 🔗 Official Links

- **COCO Website:** https://cocodataset.org/
- **COCO Download Page:** https://cocodataset.org/#download
- **COCO API:** https://github.com/cocodataset/cocoapi
- **Sapiens Model:** https://huggingface.co/facebook/sapiens-pretrain-1b-torchscript
- **This Repository:** https://github.com/alexv879/LLM_POSE_IDENTIFIER

---

## ✅ Ready to Train?

Once you have:
- [x] COCO train2017 downloaded (~18GB)
- [x] COCO val2017 downloaded (~1GB)
- [x] Annotations downloaded (~241MB)
- [x] Pretrained weights (auto-downloads)
- [x] `validate_setup.py` shows all checks passing

**You're ready!** Run:
```bash
python stages/stage1_baseline.py --config configs/stage1_config.yaml
```

Expected training time on RTX 4060:
- Phase 1 (10 epochs): 8-10 hours
- Phase 2 (100 epochs): 2-3 days

Expected performance: 82-86% AP on COCO validation set

---

**Questions?** Open an issue: https://github.com/alexv879/LLM_POSE_IDENTIFIER/issues
