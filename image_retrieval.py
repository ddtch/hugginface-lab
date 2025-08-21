# Use a pipeline as a high-level helper
from transformers import pipeline, BlipForImageTextRetrieval, AutoProcessor, AutoModelForVision2Seq
import gradio as gr
import warnings
import torch
from transformers.utils import logging
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# For image-to-text generation
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-large")

raw_image = Image.open("./assets/images/demo.jpg", 'r').convert('RGB')
text = "an image of a man and a cat in the park"

inputs = processor(images=raw_image, return_tensors="pt")

# Generate caption
out = model.generate(**inputs, max_new_tokens=100)
generated_text = processor.decode(out[0], skip_special_tokens=True)

print(f"Generated caption of image is: {generated_text}")

# For image-text matching, we need a different model
itm_processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
itm_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")

itm_inputs = itm_processor(images=raw_image, text=text, return_tensors="pt")
itm_outputs = itm_model(**itm_inputs)
itm_scores = torch.nn.functional.softmax(itm_outputs.itm_score, dim=-1)

print(f"Image and your text: `{text}` are matched with a probability of: {itm_scores[0][1]:.4f}")



