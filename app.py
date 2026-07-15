import os
import shutil
import tempfile
import subprocess
import torch
import gradio as gr

# Check if a GPU is available, default to CPU otherwise
device = "cuda" if torch.cuda.is_available() else "cpu"

def process_mri(input_file_path):
    if input_file_path is None:
        return None, "Please upload a NIfTI file (.nii.gz)."

    # 1. Create a safe temporary environment
    temp_dir = tempfile.mkdtemp()
    temp_input_dir = os.path.join(temp_dir, "input")
    temp_output_dir = os.path.join(temp_dir, "output")
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)

    try:
        # 2. Prepare filename with nnU-Net's strict '_0000.nii.gz' requirement
        base_name = "subject_temp"
        stripped_input_path = os.path.join(temp_input_dir, f"{base_name}_0000.nii.gz")

        yield None, "Step 1/2: Running HD-BET Skull Stripping on CPU (this takes 1-2 mins)..."
        
        # Build HD-BET command
        hdbet_cmd = [
            "hd-bet",
            "-i", input_file_path,
            "-o", stripped_input_path,
            "-device", "cpu",      # Force CPU for safe laptop execution
            "--disable_tta"        # Replaces old -mode/-tta flags to speed up CPU
        ]
        
        # Run HD-BET
        subprocess.run(hdbet_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        yield None, "Step 2/2: Running nnU-Net CP Segmentation (this takes 3-5 mins)..."

        # Build nnU-Net prediction command
        nnunet_cmd = [
            "nnUNetv2_predict_from_modelfolder",
            "-i", temp_input_dir,
            "-o", temp_output_dir,
            "-m", "cp_inference_model",
            "-f", "0", # Using Fold 0 for fast local CPU testing
            "-device", device,
            "-chk", "checkpoint_final.pth"
        ]

        # Run nnU-Net
        subprocess.run(nnunet_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 3. Locate the generated segmentation mask
        expected_output_mask = os.path.join(temp_output_dir, f"{base_name}.nii.gz")
        
        if os.path.exists(expected_output_mask):
            # Create a clean final path to return to user
            final_output_path = os.path.join(temp_dir, "CP_segmentation_mask.nii.gz")
            shutil.copy(expected_output_mask, final_output_path)
            yield final_output_path, "Inference completed successfully! You can download your mask below."
        else:
            yield None, "Error: The model failed to generate a segmentation mask."

    except subprocess.CalledProcessError as e:
        yield None, f"An error occurred during execution: {e.stderr.decode('utf-8', errors='ignore')}"
    except Exception as e:
        yield None, f"An unexpected error occurred: {str(e)}"

# Define the Gradio web layout
with gr.Blocks(title="Choroid Plexus Segmentation Tool") as demo:
    gr.Markdown(
        """
        # 🧠 Choroid Plexus Automated Segmentation Tool
        This app automatically performs **HD-BET skull-stripping** and **nnU-Net segmentation** of the Choroid Plexus (CP) on 3D DIR MRI scans.
        
        * **Device currently in use:** `%s` (Running on CPU is slower but works on standard laptops!)
        """ % device.upper()
    )
    
    with gr.Row():
        with gr.Column():
            input_mri = gr.File(label="Upload 3D DIR MRI (.nii.gz / .nii)", file_types=[".gz", ".nii"])
            submit_btn = gr.Button("Run Pipeline", variant="primary")
            
        with gr.Column():
            output_status = gr.Textbox(label="Processing Status", placeholder="Upload an image and click Run...", interactive=False)
            output_mask = gr.File(label="Download CP Segmentation Mask (.nii.gz)")

    # Connect the UI elements to the processing function
    submit_btn.click(
        fn=process_mri,
        inputs=[input_mri],
        outputs=[output_mask, output_status]
    )

# Launch the app locally
if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
