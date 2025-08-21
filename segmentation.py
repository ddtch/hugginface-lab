import gradio as gr
from transformers import pipeline, SamModel, SamProcessor
from transformers import AutoProcessor, AutoModelForMaskGeneration
import warnings
from transformers.utils import logging
#from helper import render_results_in_image, summarize_prediction_netural_language
from PIL import Image, ImageDraw, ImageFont
import torch
from helper import show_pipe_masks_on_image

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Force CPU usage to avoid MPS float64 issues
device = "cpu" if torch.backends.mps.is_available() else "auto"


processor = AutoProcessor.from_pretrained("Zigeng/SlimSAM-uniform-77")
model = AutoModelForMaskGeneration.from_pretrained("Zigeng/SlimSAM-uniform-77")

# Initialize pipeline with CPU device
sam_pipe = pipeline(
    "mask-generation", 
    model="Zigeng/SlimSAM-uniform-77",
    device=device,
    torch_dtype=torch.float32  # Force float32
)

raw_image = Image.open('assets/people.png')
raw_image = raw_image.resize((720, 375))

output = sam_pipe(raw_image, points_per_batch=32)

input_points = [[[720, 375]]]

inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    
predicted_masks = processor.post_process_masks(
    outputs,
    inputs["original_sizes"],
    inputs["reshaped_input_sizes"],
)




# # Debug: Print the output structure
# print("Pipeline output type:", type(output))
# if isinstance(output, dict):
#     print("Keys in output:", list(output.keys()))
#     if 'masks' in output:
#         print("Number of masks:", len(output['masks']))
#         print("Mask shape:", output['masks'][0].shape if output['masks'] else "No masks")

# # Convert format for helper function
# if isinstance(output, dict) and 'masks' in output:
#     # Create list of dictionaries with mask and score
#     formatted_output = []
#     masks = output['masks']
#     scores = output['scores'] if 'scores' in output else [1.0] * len(masks)
    
#     for i, (mask, score) in enumerate(zip(masks, scores)):
#         formatted_output.append({
#             'mask': mask,
#             'score': float(score),
#             'label': f'Object {i+1}'
#         })

# result_image.save('assets/people_segmented.png')