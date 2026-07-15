#!/bin/bash
PROJECT_DIR="$(pwd)"
mkdir -p data_out

find data_in -type f -name "*_space-t1w_DIR.nii.gz" | while read -r img; do
    basename=$(basename "$img" .nii.gz)
    echo "=========================================="
    echo "Processing: $basename"
    echo "=========================================="

    temp_input_dir=$(mktemp -d)
    
    # 1. Run HD-BET Skull Stripping
    # -device cpu : run on CPU (since we are testing locally)
    # -mode fast & -tta 0 : speeds up CPU extraction drastically
    # -save_mask 0 : we only want the stripped brain, not the binary brain mask
    echo "--> Running HD-BET Skull Stripping..."
    hd-bet \
        -i "$(realpath "$img")" \
        -o "$temp_input_dir/${basename}_0000.nii.gz" \
        -device cpu \
        -mode fast \
        -tta 0 \
        -save_mask 0

    # 2. Run nnU-Net Inference on the skull-stripped image
    echo "--> Running nnU-Net CP Segmentation..."
    nnUNetv2_predict_from_modelfolder \
        -i "$temp_input_dir" \
        -o data_out \
        -m cp_inference_model \
        -f 0 \
        -device cpu \
        -chk checkpoint_final.pth
        
    rm -r "$temp_input_dir"
    echo "--> Finished $basename!"
done