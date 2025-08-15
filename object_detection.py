import gradio as gr
from transformers import pipeline
import warnings
from transformers.utils import logging
#from helper import render_results_in_image, summarize_prediction_netural_language
from PIL import Image, ImageDraw, ImageFont
import torch

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Initialize object detection pipeline
od_pipe = pipeline("object-detection", model="facebook/detr-resnet-50")
# tts_pipe = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")
tts_pipe = pipeline("text-to-speech", model="suno/bark-small")

labels = []

# def get_pipeline_prediciton(pil_image):
#     # pipeline output from image
#     pipeline_output = od_pipe(pil_image)
    
#     # process image using pipeline output
#     processed_image = render_results_in_image(pil_image, pipeline_output)
    
#     return processed_image

def detect_objects(image):
    """
    Detect objects in an image and return annotated image
    """
    if image is None:
        return None
    
    # Run object detection
    predictions = od_pipe(image)
    print(predictions)
    
    # Create a copy of the image for annotation
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Draw bounding boxes and labels
    for prediction in predictions:
        box = prediction['box']
        label = prediction['label']
        score = prediction['score']
        labels.append(label)
        
        # Extract coordinates
        xmin = box['xmin']
        ymin = box['ymin']
        xmax = box['xmax']
        ymax = box['ymax']
        
        if score > 0.6:
          # Draw rectangle
          draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
          
          # Draw label with confidence score
          label_text = f"{label}: {score:.2f}"
          draw.text((xmin, ymin - 50), label_text, fill="green", font_size=48.0)
    
    return annotated_image

# Create Gradio interface
uplod_ui = interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="🔍 Object Detection",
    description="Upload an image to detect objects. The model will draw bounding boxes around detected objects with labels and confidence scores.",
    examples=None
)

demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(label="Upload an image to detect objects", type="pil"),
    outputs=gr.Image(label="Output image with predicted instances", type="pil"),
    title="🔍 Object Detection",
    description="Upload an image to detect objects. The model will draw bounding boxes around detected objects with labels and confidence scores.",
    examples=None
)

# Function to announce detected objects using text-to-speech
def announce_objects(image):
    if image is None:
        return "No image provided"
    
    # Run object detection
    predictions = od_pipe(image)
    
    # Filter predictions with confidence > 0.6
    detected_objects = []
    for prediction in predictions:
        if prediction['score'] > 0.6:
            detected_objects.append(prediction['label'])
    
    if not detected_objects:
        announcement = "No objects detected with high confidence."
    else:
        # Create natural language announcement
        unique_objects = list(set(detected_objects))
        if len(unique_objects) == 1:
            announcement = f"I detected a {unique_objects[0]} in the image."
        elif len(unique_objects) == 2:
            announcement = f"I detected a {unique_objects[0]} and a {unique_objects[1]} in the image."
        else:
            announcement = f"I detected {', '.join(unique_objects[:-1])}, and a {unique_objects[-1]} in the image."
    
    return announcement

# Combined function for both object detection and announcement
def detect_and_announce(image):
    if image is None:
        return None, "No image provided"
    
    # Run object detection
    predictions = od_pipe(image)
    
    # Create a copy of the image for annotation
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Filter predictions with confidence > 0.6 and create announcement
    detected_objects = []
    for prediction in predictions:
        box = prediction['box']
        label = prediction['label']
        score = prediction['score']
        
        if score > 0.6:
            detected_objects.append(label)
            
            # Extract coordinates
            xmin = box['xmin']
            ymin = box['ymin']
            xmax = box['xmax']
            ymax = box['ymax']
            
            # Draw rectangle
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
            
            # Draw label with confidence score
            label_text = f"{label}: {score:.2f}"
            draw.text((xmin, ymin - 50), label_text, fill="green", font_size=48.0)
    
    # Create announcement
    if not detected_objects:
        announcement = "No objects detected with high confidence."
    else:
        unique_objects = list(set(detected_objects))
        if len(unique_objects) == 1:
            announcement = f"I detected a {unique_objects[0]} in the image."
        elif len(unique_objects) == 2:
            announcement = f"I detected a {unique_objects[0]} and a {unique_objects[1]} in the image."
        else:
            announcement = f"I detected {', '.join(unique_objects[:-1])}, and a {unique_objects[-1]} in the image."
    
    return annotated_image, announcement

# Function to convert text to speech
def text_to_speech(text):
    if not text or text == "No image provided" or text == "No objects detected with high confidence.":
        return None
    
    print(text)
    try:
        # Generate speech from text
        audio = tts_pipe(text)
        return audio
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return None

# Enhanced interface with announcement and play button using Blocks
with gr.Blocks(title="🔍 Object Detection with Speech") as demo_with_announcement:
    gr.Markdown("# 🔍 Object Detection with Speech")
    gr.Markdown("Upload an image to detect objects. The model will draw bounding boxes around detected objects and provide a text announcement of what was found.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload an image to detect objects", type="pil")
            detect_btn = gr.Button("🔍 Detect Objects", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Output image with predicted instances", type="pil")
            announcement_block = gr.Textbox(label="Object Announcement", lines=2, interactive=False)
            play_btn = gr.Button("🔊 Play Announcement", variant="secondary")
            audio_output = gr.Audio(label="Speech Output", type="numpy")
    
    # Connect the detect button to the detection function
    detect_btn.click(
        fn=detect_and_announce,
        inputs=input_image,
        outputs=[output_image, announcement_block]  # Only 2 outputs
    )
    
    # Connect the play button to the text-to-speech function
    play_btn.click(
        fn=text_to_speech,
        inputs=announcement_block,
        outputs=audio_output
    )



if __name__ == "__main__":
    # upload_ui.launch()
    # demo.launch()
    demo_with_announcement.launch(share=True)