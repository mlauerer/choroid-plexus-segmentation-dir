#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# ==============================================================================
# CONFIGURATION & ENVIRONMENT SETUP
# ==============================================================================
DATASET_ID="999"
DATASET_NAME="Dataset${DATASET_ID}_CP_Finetune"
CONFIG="3d_fullres"      # Change if you used 2d or 3d_cascade_fullres
FOLD="0"
EPOCHS="${EPOCHS:-100}"  # Uses environment variable (e.g. from Gradio) or defaults to 100

BASE_DIR="$(pwd)"
PRETRAINED_CHECKPOINT="${BASE_DIR}/cp_inference_model/fold_0/checkpoint_final.pth"

# Define local directories for nnU-Net
export nnUNet_raw="${BASE_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${BASE_DIR}/nnUNet_preprocessed"
export nnUNet_results="${BASE_DIR}/nnUNet_results"

INPUT_DIR="${BASE_DIR}/finetune_data/images"
LABEL_DIR="${BASE_DIR}/finetune_data/labels"
TARGET_RAW_DIR="${nnUNet_raw}/${DATASET_NAME}"

echo "=========================================================================="
echo "🧠 Starting Choroid Plexus Model Fine-Tuning Workflow"
echo "=========================================================================="

# 1. Sanity Checks
if [ ! -d "$INPUT_DIR" ] || [ ! -d "$LABEL_DIR" ]; then
    echo "❌ Error: Could not find 'finetune_data/images' or 'finetune_data/labels'."
    echo "Please place raw DIR images in 'finetune_data/images' and matching masks in 'finetune_data/labels'."
    exit 1
fi

if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "❌ Error: Base weights not found at: $PRETRAINED_CHECKPOINT"
    echo "Please make sure the pre-trained model folder 'cp_inference_model' is downloaded into the project root."
    exit 1
fi

# ==============================================================================
# GPU AVAILABILITY CHECK
# ==============================================================================
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    echo "--> GPU Acceleration detected: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    DEVICE="cpu"
    echo "=========================================================================="
    echo "⚠️  WARNING: No CUDA-capable GPU detected by PyTorch!"
    echo "Fine-tuning on a CPU will be EXTREMELY slow (potentially taking days)."
    echo "=========================================================================="
    echo "Press Ctrl+C within 10 seconds to cancel, or wait to proceed on CPU..."
    sleep 10
fi

# Create target nnU-Net raw folders
mkdir -p "${TARGET_RAW_DIR}/imagesTr"
mkdir -p "${TARGET_RAW_DIR}/labelsTr"

# ==============================================================================
# STEP 1: SKULL-STRIPPING & DATA FORMATTING FOR nnU-NET
# ==============================================================================
echo "--> [Step 1/4] Preparing and skull-stripping fine-tuning cohort..."

COUNT=0
while read -r img_path; do
    filename=$(basename "$img_path")
    # Extract subject prefix (handling both .nii and .nii.gz)
    subject="${filename%%.*}"
    
    # Locate matching mask in label directory
    mask_path=$(find "$LABEL_DIR" -type f -name "${subject}*" | head -n 1)
    
    if [ -z "$mask_path" ]; then
        echo "⚠️ Warning: No matching mask found for $filename in $LABEL_DIR. Skipping."
        continue
    fi

    # Assign sequential identifier for nnU-Net (e.g., CP_001_0000.nii.gz)
    COUNT=$((COUNT + 1))
    case_id=$(printf "CP_%03d" "$COUNT")
    
    out_img="${TARGET_RAW_DIR}/imagesTr/${case_id}_0000.nii.gz"
    out_mask="${TARGET_RAW_DIR}/labelsTr/${case_id}.nii.gz"

    echo "    Processing Case ${case_id} (${filename})..."

    # 1a. Skull-strip image with HD-BET directly into imagesTr
    hd-bet -i "$img_path" -o "$out_img" -device "$DEVICE" --disable_tta > /dev/null 2>&1

    # 1b. Copy mask to labelsTr
    cp "$mask_path" "$out_mask"
done < <(find "$INPUT_DIR" -type f \( -name "*.nii.gz" -o -name "*.nii" \))

if [ "$COUNT" -eq 0 ]; then
    echo "❌ Error: No valid image-mask pairs were found to process."
    exit 1
fi

# ==============================================================================
# STEP 2: GENERATE DATASET.JSON
# ==============================================================================
echo "--> [Step 2/4] Generating nnU-Net dataset.json..."

python3 - <<EOF
import os
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

target_dir = "${TARGET_RAW_DIR}"
num_training = len(os.listdir(os.path.join(target_dir, "imagesTr")))

generate_dataset_json(
    output_folder=target_dir,
    channel_names={"0": "DIR"},
    labels={"background": 0, "choroid_plexus": 1},
    num_training_cases=num_training,
    file_ending=".nii.gz",
    dataset_name="${DATASET_NAME}",
    reference="Fine-tuning adaptation",
    description="Fine-tuning Choroid Plexus segmentation model on custom center data."
)
EOF

# ==============================================================================
# STEP 3: PLAN AND PREPROCESS
# ==============================================================================
echo "--> [Step 3/4] Running nnU-Net planning and preprocessing..."
nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -c "${CONFIG}" --verify_dataset_integrity

# ==============================================================================
# STEP 4: TRAIN ALL 5 FOLDS WITH PRE-TRAINED WEIGHTS
# ==============================================================================
echo "--> [Step 4/4] Fine-tuning 5-fold ensemble starting from pre-trained checkpoints..."
echo "Running for ${EPOCHS} epochs per fold..."

for FOLD_ID in 0 1 2 3 4; do
    FOLD_CHECKPOINT="${BASE_DIR}/cp_inference_model/fold_${FOLD_ID}/checkpoint_final.pth"
    
    if [ ! -f "$FOLD_CHECKPOINT" ]; then
        echo "⚠️ Checkpoint for fold_${FOLD_ID} not found at ${FOLD_CHECKPOINT}. Skipping fold ${FOLD_ID}."
        continue
    fi

    echo "=========================================="
    echo "--> Fine-tuning Fold ${FOLD_ID}/4..."
    echo "=========================================="

    nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD_ID}" \
        -tr nnUNetTrainer \
        -pretrained_weights "${FOLD_CHECKPOINT}" \
        --num_epochs "${EPOCHS}"
done

echo "=========================================================================="
echo "✅ Fine-tuning complete for all 5 folds!"
echo "Your newly adapted model weights are saved in: ${nnUNet_results}/${DATASET_NAME}/"
echo "=========================================================================="