import os
import shutil
import tempfile
import subprocess
import torch
import gradio as gr

# Check if a GPU is available, default to CPU otherwise
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 1. INFERENCE LOGIC
# ==============================================================================
def process_mri(input_file_paths):
    if not input_file_paths:
        return None, "Please upload at least one NIfTI file (.nii.gz / .nii)."

    session_temp_dir = tempfile.mkdtemp()
    completed_mask_paths = []
    total_files = len(input_file_paths)

    for index, file_obj in enumerate(input_file_paths):
        input_file_path = file_obj.name
        original_filename = os.path.basename(input_file_path)
        
        if original_filename.endswith(".nii.gz"):
            subject_id = original_filename[:-7]
        elif original_filename.endswith(".nii"):
            subject_id = original_filename[:-4]
        else:
            subject_id = f"subject_{index + 1}"

        progress_prefix = f"[{index + 1}/{total_files}] {original_filename}"

        temp_input_dir = os.path.join(session_temp_dir, f"input_{index}")
        temp_output_dir = os.path.join(session_temp_dir, f"output_{index}")
        os.makedirs(temp_input_dir, exist_ok=True)
        os.makedirs(temp_output_dir, exist_ok=True)

        try:
            internal_base = "subject_temp"
            stripped_input_path = os.path.join(temp_input_dir, f"{internal_base}_0000.nii.gz")

            yield completed_mask_paths, f"{progress_prefix} - Step 1/2: Running HD-BET Skull Stripping..."
            
            hdbet_cmd = [
                "hd-bet",
                "-i", input_file_path,
                "-o", stripped_input_path,
                "-device", device,
                "--disable_tta"
            ]
            subprocess.run(hdbet_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            yield completed_mask_paths, f"{progress_prefix} - Step 2/2: Running nnU-Net CP Segmentation (5-fold ensemble)..."

            # Updated -f flag to run full 5-fold ensemble
            nnunet_cmd = [
                "nnUNetv2_predict_from_modelfolder",
                "-i", temp_input_dir,
                "-o", temp_output_dir,
                "-m", "cp_inference_model",
                "-f", "0", "1", "2", "3", "4",
                "-device", device,
                "-chk", "checkpoint_final.pth"
            ]
            subprocess.run(nnunet_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            expected_output_mask = os.path.join(temp_output_dir, f"{internal_base}.nii.gz")
            if os.path.exists(expected_output_mask):
                final_mask_name = f"{subject_id}_CP_mask.nii.gz"
                final_output_path = os.path.join(session_temp_dir, final_mask_name)
                shutil.copy(expected_output_mask, final_output_path)
                completed_mask_paths.append(final_output_path)
            else:
                yield completed_mask_paths, f"Warning: Failed to generate mask for {original_filename}."

        except subprocess.CalledProcessError as e:
            yield completed_mask_paths, f"Error on {original_filename}: {e.stderr.decode('utf-8', errors='ignore')}"
        except Exception as e:
            yield completed_mask_paths, f"Unexpected error on {original_filename}: {str(e)}"

    yield completed_mask_paths, f"Successfully processed all {total_files} files! Download them below."


# ==============================================================================
# 2. FINE-TUNING LOGIC (With GPU Safety Guardrail)
# ==============================================================================
def run_finetuning_gui(image_files, mask_files, epochs, force_cpu):
    if device != "cuda" and not force_cpu:
        yield (
            "⚠️ WARNING: No CUDA-enabled GPU was detected on this system!\n\n"
            "Fine-tuning a 3D nnU-Net model on a CPU is extremely slow and can take several days.\n"
            "If you still wish to proceed on CPU, check the 'Force CPU Execution' box and click Start again."
        )
        return

    if not image_files or not mask_files:
        yield "❌ Error: Please upload both DIR image files and matching mask files."
        return

    if len(image_files) != len(mask_files):
        yield f"❌ Error: Mismatch in file count. You provided {len(image_files)} images but {len(mask_files)} masks."
        return

    project_root = os.path.abspath(os.path.dirname(__file__))
    finetune_data_dir = os.path.join(project_root, "finetune_data")
    img_target_dir = os.path.join(finetune_data_dir, "images")
    lbl_target_dir = os.path.join(finetune_data_dir, "labels")

    shutil.rmtree(img_target_dir, ignore_errors=True)
    shutil.rmtree(lbl_target_dir, ignore_errors=True)
    os.makedirs(img_target_dir, exist_ok=True)
    os.makedirs(lbl_target_dir, exist_ok=True)

    yield "--> Stage 1/2: Staging images and masks into finetune_data/..."

    sorted_images = sorted(image_files, key=lambda x: os.path.basename(x.name))
    sorted_masks = sorted(mask_files, key=lambda x: os.path.basename(x.name))

    for img_obj, mask_obj in zip(sorted_images, sorted_masks):
        shutil.copy(img_obj.name, os.path.join(img_target_dir, os.path.basename(img_obj.name)))
        shutil.copy(mask_obj.name, os.path.join(lbl_target_dir, os.path.basename(mask_obj.name)))

    execution_device = "CUDA GPU" if device == "cuda" else "CPU (Forced)"
    yield f"--> Stage 2/2: Launching run_finetuning.sh for {epochs} epochs per fold on [{execution_device}]...\n"

    env = os.environ.copy()
    env["EPOCHS"] = str(int(epochs))

    script_path = os.path.join(project_root, "run_finetuning.sh")
    if not os.path.exists(script_path):
        yield "❌ Error: Could not find 'run_finetuning.sh' script in working directory."
        return

    process = subprocess.Popen(
        ["bash", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=project_root,
        env=env
    )

    log_output = "================ Start of Fine-Tuning Execution ================\n"
    for line in iter(process.stdout.readline, ''):
        log_output += line
        yield log_output

    process.wait()
    if process.returncode == 0:
        log_output += "\n========================================================\n"
        log_output += "✅ Fine-Tuning process completed successfully!"
    else:
        log_output += f"\n❌ Script exited with error code {process.returncode}."

    yield log_output


# ==============================================================================
# 3. GRADIO INTERFACE LAYOUT
# ==============================================================================
with gr.Blocks(title="Choroid Plexus Segmentation & Fine-Tuning Suite") as demo:
    gr.Markdown(
        f"""
        # 🧠 Choroid Plexus (CP) Segmentation & Adaptation Suite
        * **Active Device Acceleration:** `{device.upper()}`
        """
    )

    with gr.Tabs():
        # TAB 1: INFERENCE
        with gr.TabItem("Inference (Apply Model)"):
            gr.Markdown(
                """
                ### Run CP Segmentation on New DIR Scans
                Upload one or multiple 3D DIR MRI scans (.nii / .nii.gz). The pipeline will automatically skull-strip each image with HD-BET and extract the Choroid Plexus mask.
                """
            )
            with gr.Row():
                with gr.Column():
                    input_mri = gr.File(
                        label="Upload 3D DIR MRIs", 
                        file_types=[".gz", ".nii"], 
                        file_count="multiple"
                    )
                    submit_btn = gr.Button("Run Inference Pipeline", variant="primary")
                    
                with gr.Column():
                    output_status = gr.Textbox(label="Processing Status", placeholder="Awaiting upload...", interactive=False)
                    output_masks = gr.File(label="Download CP Segmentation Masks", file_count="multiple")

            submit_btn.click(
                fn=process_mri,
                inputs=[input_mri],
                outputs=[output_masks, output_status]
            )

        # TAB 2: FINE-TUNING
        with gr.TabItem("Fine-Tuning (Adapt Model)"):
            gr.Markdown(
                """
                ### Fine-Tune Model with Local Annotations
                If the out-of-the-box model needs adaptation for your specific MRI scanner, upload **10–15 subject DIR scans** along with their **corresponding ground-truth manual CP masks**.
                """
            )
            with gr.Row():
                with gr.Column():
                    ft_images = gr.File(
                        label="1. Upload Training DIR MRIs (.nii.gz)", 
                        file_types=[".gz", ".nii"], 
                        file_count="multiple"
                    )
                    ft_masks = gr.File(
                        label="2. Upload Ground-Truth Masks (.nii.gz)", 
                        file_types=[".gz", ".nii"], 
                        file_count="multiple"
                    )
                    ft_epochs = gr.Slider(
                        minimum=20, 
                        maximum=300, 
                        value=100, 
                        step=10, 
                        label="Fine-Tuning Epochs",
                        info="100 epochs is typically ideal for transfer learning on 10-15 cases."
                    )
                    force_cpu_cb = gr.Checkbox(
                        label="Force CPU Execution",
                        value=False,
                        info="Check this ONLY if you do not have a GPU and accept extremely slow runtime."
                    )
                    start_ft_btn = gr.Button("Start Fine-Tuning Process", variant="stop")

                with gr.Column():
                    ft_logs = gr.Textbox(
                        label="Live Execution Terminal Log", 
                        placeholder="Log output from run_finetuning.sh will stream here...", 
                        lines=20,
                        interactive=False
                    )

            start_ft_btn.click(
                fn=run_finetuning_gui,
                inputs=[ft_images, ft_masks, ft_epochs, force_cpu_cb],
                outputs=[ft_logs]
            )

if __name__ == "__main__":
    # Binding to 0.0.0.0 enables Docker container access and remote hosting
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)