#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

PROJECT_DIR="$(pwd)"
DATA_IN="${PROJECT_DIR}/data_in"
DATA_OUT="${PROJECT_DIR}/data_out"

# Check if data_in contains NIfTI files
if [ ! -d "$DATA_IN" ] || [ -z "$(find "$DATA_IN" -type f \( -name "*.nii.gz" -o -name "*.nii" \))" ]; then
    echo "❌ No NIfTI files (.nii / .nii.gz) found in 'data_in/'."
    echo "Please copy your 3D DIR MRI scans into 'data_in/' and rerun this script."
    exit 1
fi

# Create data_out directory only once files are confirmed present
mkdir -p "$DATA_OUT"

# Detect GPU or CPU execution
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    echo "--> GPU Acceleration detected. Running inference on GPU..."
else
    DEVICE="cpu"
    echo "⚠️  No GPU detected. Running inference on CPU (this may take longer)..."
fi

echo "=========================================================================="
echo "🧠 Starting Choroid Plexus Segmentation Pipeline"
echo "=========================================================================="

# Process all .nii.gz and .nii files in data_in/
while read -r img; do
    filename=$(basename "$img")
    
    # Extract subject ID handling both .nii.gz and .nii extensions
    if [[ "$filename" == *.nii.gz ]]; then
        basename="${filename%.nii.gz}"
    else
        basename="${filename%.nii}"
    fi

    echo "=========================================="
    echo "Processing: $basename"
    echo "=========================================="

    temp_input_dir=$(mktemp -d)

    # Step 1: HD-BET Skull Stripping
    echo "--> Running HD-BET Skull Stripping..."
    hd-bet \
        -i "$(realpath "$img")" \
        -o "$temp_input_dir/${basename}_0000.nii.gz" \
        -device "$DEVICE" \
        --disable_tta > /dev/null

    # Step 2: nnU-Net CP Segmentation (Ensemble of folds 0 through 4)
    echo "--> Running nnU-Net CP Segmentation (5-fold ensemble)..."
    nnUNetv2_predict_from_modelfolder \
        -i "$temp_input_dir" \
        -o "$DATA_OUT" \
        -m cp_inference_model \
        -f 0 1 2 3 4 \
        -device "$DEVICE" \
        -chk checkpoint_final.pth > /dev/null

    # Rename output to a clean, descriptive mask filename
    if [ -f "$DATA_OUT/${basename}.nii.gz" ]; then
        mv "$DATA_OUT/${basename}.nii.gz" "$DATA_OUT/${basename}_CP_mask.nii.gz"
    fi

    rm -rf "$temp_input_dir"
    echo "--> Successfully generated: data_out/${basename}_CP_mask.nii.gz"
done < <(find "$DATA_IN" -type f \( -name "*.nii.gz" -o -name "*.nii" \))

echo "=========================================================================="
echo "✅ All scans processed successfully! Results saved in 'data_out/'."
echo "=========================================================================="