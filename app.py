import os
import shutil
import tempfile
import subprocess
import torch
import gradio as gr

# Check if a GPU is available, default to CPU otherwise
device = "cuda" if torch.cuda.is_available() else "cpu"

def process_mri(input_file_paths):
    if not input_file_paths:
        return None, "Please upload at least one NIfTI file (.nii.gz)."

    # 1. Create a safe temporary directory to hold all output masks
    session_temp_dir = tempfile.mkdtemp()
    completed_mask_paths = []
    
    total_files = len(input_file_paths)

    # Loop through each uploaded scan
    for index, file_obj in enumerate(input_file_paths):
        # Gradio supplies file objects; extract the actual path
        input_file_path = file_obj.name
        original_filename = os.path.basename(input_file_path)
        
        # Extract subject ID for the filename
        if original_filename.endswith(".nii.gz"):
            subject_id = original_filename[:-7]
        elif original_filename.endswith(".nii"):
            subject_id = original_filename[:-4]
        else:
            subject_id = f"subject_{index + 1}"

        progress_prefix = f"[{index + 1}/{total_files}] {original_filename}"

        # Create temporary input/output folders *per subject* to avoid nnU-Net collisions
        temp_input_dir = os.path.join(session_temp_dir, f"input_{index}")
        temp_output_dir = os.path.join(session_temp_dir, f"output_{index}")
        os.makedirs(temp_input_dir, exist_ok=True)
        os.makedirs(temp_output_dir, exist_ok=True)

        try:
            # Prepare internal naming for nnU-Net's strict requirement
            internal_base = "subject_temp"
            stripped_input_path = os.path.join(temp_input_dir, f"{internal_base}_0000.nii.gz")

            yield completed_mask_paths, f"{progress_prefix} - Step 1/2: Running HD-BET Skull Stripping..."
            
            # Run HD-BET
            hdbet_cmd = [
                "hd-bet",
                "-i", input_file_path,
                "-o", stripped_input_path,
                "-device", "cpu",
                "--disable_tta"
            ]
            subprocess.run(hdbet_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            yield completed_mask_paths, f"{progress_prefix} - Step 2/2: Running nnU-Net CP Segmentation..."

            # Run nnU-Net
            nnunet_cmd = [
                "nnUNetv2_predict_from_modelfolder",
                "-i", temp_input_dir,
                "-o", temp_output_dir,
                "-m", "cp_inference_model",
                "-f", "0",
                "-device", device,
                "-chk", "checkpoint_final.pth"
            ]
            subprocess.run(nnunet_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Locate and rename the output mask
            expected_output_mask = os.path.join(temp_output_dir, f"{internal_base}.nii.gz")
            if os.path.exists(expected_output_mask):
                final_mask_name = f"{subject_id}_CP_mask.nii.gz"
                final_output_path = os.path.join(session_temp_dir, final_mask_name)
                shutil.copy(expected_output_mask, final_output_path)
                
                # Add this finished mask to our download list
                completed_mask_paths.append(final_output_path)
            else:
                yield completed_mask_paths, f"Warning: Failed to generate mask for {original_filename}."

        except subprocess.CalledProcessError as e:
            yield completed_mask_paths, f"Error on {original_filename}: {e.stderr.decode('utf-8', errors='ignore')}"
        except Exception as e:
            yield completed_mask_paths, f"Unexpected error on {original_filename}: {str(e)}"

    yield completed_mask_paths, f"Successfully processed all {total_files} files! Download them below."


# Define the Gradio web layout
with gr.Blocks(title="Choroid Plexus Segmentation Tool") as demo:
    gr.Markdown(
        """
        # 🧠 Choroid Plexus Automated Segmentation Tool
        Upload one or **multiple** 3D DIR MRI scans. The pipeline will automatically:
        1. Run **HD-BET skull-stripping** on each scan.
        2. Perform **nnU-Net segmentation** of the Choroid Plexus (CP).
        3. Provide individual download links for all generated masks.
        
        * **Running on:** `%s`
        """ % device.upper()
    )
    
    with gr.Row():
        with gr.Column():
            # Added file_count="multiple" to allow batch uploads!
            input_mri = gr.File(
                label="Upload 3D DIR MRIs (.nii.gz / .nii)", 
                file_types=[".gz", ".nii"], 
                file_count="multiple"
            )
            submit_btn = gr.Button("Run Batch Pipeline", variant="primary")
            
        with gr.Column():
            output_status = gr.Textbox(label="Processing Status", placeholder="Upload images and click Run...", interactive=False)
            # Added file_count="multiple" so Gradio displays a list of output files to download
            output_masks = gr.File(label="Download CP Segmentation Masks", file_count="multiple")

    # Connect the UI elements to the processing function
    submit_btn.click(
        fn=process_mri,
        inputs=[input_mri],
        outputs=[output_masks, output_status]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)