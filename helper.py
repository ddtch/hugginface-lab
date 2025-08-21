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

def show_pipe_masks_on_image(pil_image, pipeline_output, alpha=0.5, colors=None):
    """
    Overlay segmentation masks on the input image
    
    Args:
        pil_image: PIL Image object
        pipeline_output: Output from segmentation pipeline
        alpha: Transparency of the mask overlay (0.0 to 1.0)
        colors: List of colors for different classes (optional)
    
    Returns:
        PIL Image with masks overlaid
    """
    import numpy as np
    
    # Default colors for different classes
    if colors is None:
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (255, 192, 203), # Pink
            (0, 128, 0),    # Dark Green
        ]
    
    # Convert PIL image to numpy array
    image_array = np.array(pil_image)
    
    # Create a copy for overlay
    overlay_image = image_array.copy()
    
    # Handle different output formats
    if isinstance(pipeline_output, str):
        # If output is a string, return original image
        print("Warning: Pipeline output is a string, returning original image")
        return pil_image
    
    # Process each segmentation result
    for i, result in enumerate(pipeline_output):
        # Handle different result formats
        if isinstance(result, dict) and 'mask' in result:
            mask = result['mask']
            label = result.get('label', f'Class {i}')
            score = result.get('score', 0.0)
        elif hasattr(result, 'mask'):
            # If result is an object with mask attribute
            mask = result.mask
            label = getattr(result, 'label', f'Class {i}')
            score = getattr(result, 'score', 0.0)
        else:
            # Skip if result doesn't have mask
            continue
        
        # Convert tensor to numpy array if needed and ensure float32
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy().astype(np.float32)
        elif isinstance(mask, np.ndarray):
            mask = mask.astype(np.float32)
        else:
            # Skip if mask is not a tensor or numpy array
            continue
        
        # Get color for this class
        color = colors[i % len(colors)]
        
        # Apply mask overlay
        if mask.ndim == 2:  # 2D mask
            # Create colored mask
            colored_mask = np.zeros_like(image_array, dtype=np.float32)
            colored_mask[mask > 0.5] = color  # Threshold the mask
            
            # Blend with original image
            overlay_image = (1 - alpha) * overlay_image.astype(np.float32) + alpha * colored_mask
            
        elif mask.ndim == 3:  # 3D mask (multiple channels)
            # Use the first channel if it's a multi-channel mask
            mask_2d = mask[:, :, 0] if mask.shape[2] == 1 else mask.any(axis=2)
            colored_mask = np.zeros_like(image_array, dtype=np.float32)
            colored_mask[mask_2d > 0.5] = color  # Threshold the mask
            
            # Blend with original image
            overlay_image = (1 - alpha) * overlay_image.astype(np.float32) + alpha * colored_mask
    
    # Convert back to PIL Image
    result_image = Image.fromarray(overlay_image.astype(np.uint8))
    
    # Add labels and scores as text overlay
    draw = ImageDraw.Draw(result_image)
    
    for i, result in enumerate(pipeline_output):
        # Handle different result formats
        if isinstance(result, dict) and 'mask' in result:
            mask = result['mask']
            label = result.get('label', f'Class {i}')
            score = result.get('score', 0.0)
        elif hasattr(result, 'mask'):
            mask = result.mask
            label = getattr(result, 'label', f'Class {i}')
            score = getattr(result, 'score', 0.0)
        else:
            continue
        
        # Convert tensor to numpy array if needed
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy().astype(np.float32)
        elif isinstance(mask, np.ndarray):
            mask = mask.astype(np.float32)
        else:
            continue
        
        if mask.ndim == 3:
            mask = mask[:, :, 0] if mask.shape[2] == 1 else mask.any(axis=2)
        
        # Find the top-left corner of the mask
        rows, cols = np.where(mask > 0.5)  # Threshold the mask
        if len(rows) > 0 and len(cols) > 0:
            x, y = cols.min(), rows.min()
            
            # Draw label with background
            label_text = f"{label}: {score:.2f}"
            text_bbox = draw.textbbox((x, y), label_text)
            
            # Draw background rectangle
            draw.rectangle(text_bbox, fill="black", outline="white")
            
            # Draw text
            draw.text((x, y), label_text, fill="white")
    
    return result_image

def create_segmentation_summary(pipeline_output):
    """
    Create a summary of segmentation results
    
    Args:
        pipeline_output: Output from segmentation pipeline
    
    Returns:
        List of detected objects with their areas and confidence scores
    """
    import numpy as np
    summary = []
    
    for i, result in enumerate(pipeline_output):
        if 'mask' in result:
            mask = result['mask']
            label = result.get('label', f'Class {i}')
            score = result.get('score', 0.0)
            
            # Convert tensor to numpy array if needed and ensure float32
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy().astype(np.float32)
            elif isinstance(mask, np.ndarray):
                mask = mask.astype(np.float32)
            
            # Calculate area (number of pixels)
            if mask.ndim == 2:
                area = np.sum(mask > 0.5)  # Threshold the mask
            elif mask.ndim == 3:
                area = np.sum(mask.any(axis=2) > 0.5)  # Threshold the mask
            else:
                area = 0
            
            summary.append({
                'label': label,
                'score': score,
                'area': int(area),
                'area_percentage': float((area / (mask.shape[0] * mask.shape[1])) * 100)
            })
    
    return summary
