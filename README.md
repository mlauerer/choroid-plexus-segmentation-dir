# 🧠 Automated Choroid Plexus Segmentation on Double Inversion Recovery (DIR) MRI

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![nnU-Net](https://img.shields.io/badge/nnU--Net-v2-blue.svg)](https://github.com/MIC-DKFZ/nnUNet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official repository for automated segmentation of the **Choroid Plexus (CP)** from 3D Double Inversion Recovery (DIR) MRI sequences using an optimized 5-fold **nnU-Net v2** ensemble architecture.

This repository offers two distinct execution workflows:
1. **Interactive Web GUI (Gradio):** A user-friendly, "plug-and-play" interface ideal for clinical researchers and non-technical users.
2. **Command-Line Interface (CLI):** A fast, scriptable workflow designed for batch processing on computing clusters or workstations.

In addition to standard inference, an integrated **fine-tuning pipeline** is provided to adapt the pre-trained weights to custom center data or scanner protocols using a small cohort of manual annotations (10–15 subjects).

---

## 📁 Repository Overview

```text
├── cp_inference_model/       # Pre-trained 5-fold ensemble weights (Download separately)
├── data_in/                  # Place raw 3D DIR MRIs here (.nii / .nii.gz)
├── data_out/                 # Generated CP segmentation masks saved here
├── finetune_data/            # Training data folder for model adaptation
│   ├── images/               # Raw 3D DIR MRIs
│   └── labels/               # Ground-truth manual CP masks
├── run_prediction.sh         # CLI script for batch inference
├── run_finetuning.sh         # CLI script for model adaptation/fine-tuning
├── app.py                    # Gradio Web Interface
├── Dockerfile                # Containerized environment definition
└── requirements.txt          # Python dependencies
```

---

## 📥 Downloading Model Weights

Because nnU-Net model folders exceed standard GitHub file limits, the pre-trained weights are hosted as a compressed asset on GitHub Releases.

Run the following commands in the root of the repository to retrieve and extract the model weights:

```bash
# Download the pre-trained model weights
wget [https://github.com/mlauerer/choroid-plexus-segmentation-dir/releases/download/v1.0.0/cp_inference_model.zip](https://github.com/mlauerer/choroid-plexus-segmentation-dir/releases/download/v1.0.0/cp_inference_model.zip)

# Extract to repository root
unzip cp_inference_model.zip
rm cp_inference_model.zip
```

Ensure that the `cp_inference_model/` folder sits directly inside your repository directory.

---

## ⚙️ Installation

### Option 1: Native Virtual Environment (`venv`)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mlauerer/choroid-plexus-segmentation-dir.git](https://github.com/mlauerer/choroid-plexus-segmentation-dir.git)
   cd choroid-plexus-segmentation-dir
   ```

2. **Set up virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install PyTorch with CUDA support:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
   ```

4. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

### Option 2: Docker Container

To run the pipeline inside an isolated environment with pre-configured dependencies:

1. **Build the container image:**
   ```bash
   docker build -t choroid-plexus-seg .
   ```

2. **Launch the Web GUI via Docker:**
   ```bash
   docker run --gpus all -p 7860:7860 choroid-plexus-seg
   ```
   Access the GUI at `http://localhost:7860` in your web browser.

---

## 🖥️ Usage & Workflows

### Workflow A: Interactive Web Interface (Plug-and-Play)

The web GUI provides an accessible interface to process scans or perform transfer learning without using the command line.

1. **Launch the web application:**
   ```bash
   python3 app.py
   ```
2. Open `http://127.0.0.1:7860` in your web browser.
3. **Inference Tab:** Drag-and-drop one or multiple 3D DIR scans (`.nii` or `.nii.gz`). The interface handles HD-BET skull stripping and nnU-Net segmentation, then provides direct download links for the generated CP masks.
4. **Fine-Tuning Tab:** Upload matching DIR scans and ground-truth manual masks, set training epochs, and monitor live terminal training logs directly within the browser.

---

### Workflow B: Command-Line Interface (CLI)

For high-throughput batch processing on local workstations or HPC environments:

1. Copy your 3D DIR MRI scans (`.nii` or `.nii.gz`) into the `data_in/` folder.
2. Run the automated inference script:
   ```bash
   ./run_prediction.sh
   ```
3. The script will automatically detect available GPU acceleration, perform HD-BET skull stripping, run the 5-fold nnU-Net ensemble, and write the output masks to `data_out/` as `<subject>_CP_mask.nii.gz`.

---

## 🎯 Fine-Tuning / Model Adaptation (Optional)

If out-of-the-box performance is insufficient due to center-specific acquisition parameters or scanner differences, the pre-trained model can be adapted using **10–15 locally annotated cases**.

### CLI Fine-Tuning Execution:
1. Place raw training DIR MRIs into `finetune_data/images/`.
2. Place corresponding ground-truth manual CP masks into `finetune_data/labels/`.
3. Execute the adaptation script:
   ```bash
   ./run_finetuning.sh
   ```
*(Note: Fine-tuning requires a CUDA-capable GPU. The script will sequentially perform transfer learning across all 5 folds to preserve ensemble performance).*

---

## 📄 Citation

If you use this tool or model in your research, please cite our paper:

```bibtex
@article{lauerer2026choroid,
  title={Improved segmentation of the choroid plexus using double inversion recovery MRI},
  author={Lauerer, M. et al.},
  journal={medRxiv},
  year={2026}
}
```