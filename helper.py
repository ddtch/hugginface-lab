# helper.py
from PIL import Image, ImageDraw, ImageFont

def render_results_in_image(pil_image, pipeline_output):
    """
    Render object detection results on the image
    """
    annotated_image = pil_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    for prediction in pipeline_output:
        box = prediction['box']
        label = prediction['label']
        score = prediction['score']
        
        if score > 0.6:
            # Draw bounding box
            draw.rectangle([box['xmin'], box['ymin'], box['xmax'], box['ymax']], 
                         outline="red", width=2)
            
            # Draw label
            label_text = f"{label}: {score:.2f}"
            draw.text((box['xmin'], box['ymin'] - 20), label_text, fill="red")
    
    return annotated_image

def summarize_prediction_netural_language(pipeline_output):
    """
    Summarize object detection results in a neutral language
    """
    summary = []
    for prediction in pipeline_output:
        label = prediction['label']
        score = prediction['score']
        
        if score > 0.6:
            summary.append(f"{label} with a confidence of {score:.2f}")
            
    return summary
