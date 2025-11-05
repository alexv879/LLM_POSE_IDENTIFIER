# Setup script for training preparation
# This script downloads pretrained weights and COCO dataset

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check GPU
Write-Host "1. Checking GPU..." -ForegroundColor Yellow
nvidia-smi --query-gpu=name, memory.total, memory.free --format=csv
Write-Host ""

# Warning about GPU memory
Write-Host "WARNING: GTX 1650 has only 4GB VRAM" -ForegroundColor Red
Write-Host "Configuration has been adjusted:" -ForegroundColor Yellow
Write-Host "  - Batch size: 32 -> 2 (with gradient accumulation x16)" -ForegroundColor Yellow
Write-Host "  - Input resolution: 512x384 -> 384x288" -ForegroundColor Yellow
Write-Host "  - Num workers: 4 -> 2" -ForegroundColor Yellow
Write-Host ""

# Create directories
Write-Host "2. Creating directories..." -ForegroundColor Yellow
$dirs = @(
    "data/coco/images/train2017",
    "data/coco/images/val2017",
    "data/pretrained/sapiens_1b",
    "checkpoints/stage1",
    "logs"
)

foreach ($dir in $dirs) {
    $fullPath = Join-Path (Get-Location) $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Green
    }
    else {
        Write-Host "  Exists: $dir" -ForegroundColor Gray
    }
}
Write-Host ""

# Check COCO annotations
Write-Host "3. Checking COCO annotations..." -ForegroundColor Yellow
$trainAnnot = "data/coco/annotations/person_keypoints_train2017.json"
$valAnnot = "data/coco/annotations/person_keypoints_val2017.json"

if (Test-Path $trainAnnot) {
    Write-Host "  Found: $trainAnnot" -ForegroundColor Green
}
else {
    Write-Host "  Missing: $trainAnnot" -ForegroundColor Red
}

if (Test-Path $valAnnot) {
    Write-Host "  Found: $valAnnot" -ForegroundColor Green
}
else {
    Write-Host "  Missing: $valAnnot" -ForegroundColor Red
}
Write-Host ""

# Check COCO images
Write-Host "4. Checking COCO images..." -ForegroundColor Yellow
$trainImages = "data/coco/images/train2017"
$valImages = "data/coco/images/val2017"

$trainCount = 0
$valCount = 0

if (Test-Path $trainImages) {
    $trainCount = (Get-ChildItem $trainImages -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
    Write-Host "  Train images: $trainCount" -ForegroundColor $(if ($trainCount -gt 0) { "Green" } else { "Red" })
}
else {
    Write-Host "  Train images directory missing" -ForegroundColor Red
}

if (Test-Path $valImages) {
    $valCount = (Get-ChildItem $valImages -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
    Write-Host "  Val images: $valCount" -ForegroundColor $(if ($valCount -gt 0) { "Green" } else { "Red" })
}
else {
    Write-Host "  Val images directory missing" -ForegroundColor Red
}
Write-Host ""

# Download COCO dataset instructions
if ($trainCount -eq 0 -or $valCount -eq 0) {
    Write-Host "COCO Dataset Not Found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "To download COCO 2017 dataset:" -ForegroundColor Yellow
    Write-Host "1. Download train2017 images (18GB):" -ForegroundColor Cyan
    Write-Host "   http://images.cocodataset.org/zips/train2017.zip" -ForegroundColor White
    Write-Host ""
    Write-Host "2. Download val2017 images (1GB):" -ForegroundColor Cyan
    Write-Host "   http://images.cocodataset.org/zips/val2017.zip" -ForegroundColor White
    Write-Host ""
    Write-Host "3. Extract to data/coco/images/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Or use Python script:" -ForegroundColor Yellow
    Write-Host "   python scripts/download_coco.py" -ForegroundColor White
    Write-Host ""
}

# Check for pretrained weights
Write-Host "5. Checking pretrained weights..." -ForegroundColor Yellow
$pretrainedPath = "data/pretrained/sapiens_1b"

# Check if HuggingFace CLI is available
$hfCliAvailable = $null -ne (Get-Command "huggingface-cli" -ErrorAction SilentlyContinue)

if ($hfCliAvailable) {
    Write-Host "  HuggingFace CLI found" -ForegroundColor Green
    
    # Check if model is already downloaded
    $modelFiles = Get-ChildItem $pretrainedPath -Filter "*.bin" -ErrorAction SilentlyContinue
    if ($modelFiles.Count -gt 0) {
        Write-Host "  Pretrained weights found: $($modelFiles.Count) files" -ForegroundColor Green
    }
    else {
        Write-Host "  Downloading Sapiens-1B from HuggingFace..." -ForegroundColor Cyan
        Write-Host "  This may take several minutes (~2-3GB download)..." -ForegroundColor Yellow
        
        # Download the model
        huggingface-cli download facebook/sapiens-1b-pretrain --local-dir $pretrainedPath
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Download complete!" -ForegroundColor Green
        }
        else {
            Write-Host "  Download failed. Please download manually." -ForegroundColor Red
        }
    }
}
else {
    Write-Host "  HuggingFace CLI not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "To install HuggingFace CLI:" -ForegroundColor Yellow
    Write-Host "   pip install huggingface_hub" -ForegroundColor White
    Write-Host ""
    Write-Host "Then download Sapiens-1B weights:" -ForegroundColor Yellow
    Write-Host "   huggingface-cli download facebook/sapiens-1b-pretrain --local-dir data/pretrained/sapiens_1b" -ForegroundColor White
    Write-Host ""
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$issues = @()

if ($trainCount -eq 0) { $issues += "Missing COCO train images" }
if ($valCount -eq 0) { $issues += "Missing COCO val images" }
if (-not (Test-Path $trainAnnot)) { $issues += "Missing train annotations" }
if (-not (Test-Path $valAnnot)) { $issues += "Missing val annotations" }

$modelExists = (Get-ChildItem $pretrainedPath -Filter "*.bin" -ErrorAction SilentlyContinue).Count -gt 0
if (-not $modelExists) { $issues += "Missing pretrained weights" }

if ($issues.Count -eq 0) {
    Write-Host "✅ All prerequisites ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To start training:" -ForegroundColor Yellow
    Write-Host "   python train_stage1.py --config configs/stage1_config.yaml" -ForegroundColor White
    Write-Host ""
    Write-Host "Expected training time (10 epochs Phase 1):" -ForegroundColor Yellow
    Write-Host "   - GTX 1650 (4GB): ~8-12 hours" -ForegroundColor White
    Write-Host "   - With gradient accumulation: slower but same quality" -ForegroundColor White
}
else {
    Write-Host "⚠️  Issues found:" -ForegroundColor Red
    foreach ($issue in $issues) {
        Write-Host "   - $issue" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Please resolve these issues before training." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Configuration file: configs/stage1_config.yaml" -ForegroundColor Cyan
Write-Host "Hardware optimization for GTX 1650 (4GB) applied" -ForegroundColor Cyan
Write-Host ""
