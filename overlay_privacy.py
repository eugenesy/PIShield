import cv2
import numpy as np
import os
from typing import Dict, Union

def overlay_images(original_path: str, overlay_path: str) -> Dict[str, Union[str, bool]]:
    """
    Overlay a PNG image (with transparency) over the original image

    Args:
        original_path (str): Path to the original image
        overlay_path (str): Path to the PNG overlay image with transparency

    Returns:
        dict: Dictionary containing:
            - success (bool): Whether processing was successful
            - output_path (str): Path to processed image if successful, None if failed
            - error (str): Error message if processing failed, None if successful
    """
    result = {
        'success': False,
        'output_path': None,
        'error': None
    }

    try:
        # Read the original image
        original = cv2.imread(original_path)
        if original is None:
            result['error'] = f"Could not load original image from {original_path}"
            return result

        # Read the overlay image with alpha channel
        overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            result['error'] = f"Could not load overlay image from {overlay_path}"
            return result

        # Ensure overlay has an alpha channel
        if overlay.shape[-1] != 4:
            result['error'] = "Overlay image must be a PNG with transparency (alpha channel)"
            return result

        # Resize overlay to match original image size if needed
        if original.shape[:2] != overlay.shape[:2]:
            overlay = cv2.resize(overlay, (original.shape[1], original.shape[0]))

        # Split the overlay into color and alpha channels
        overlay_rgb = overlay[:, :, :3]
        alpha = overlay[:, :, 3] / 255.0  # Normalize alpha to 0-1

        # Create alpha channels for broadcasting
        alpha_3d = np.stack([alpha] * 3, axis=-1)

        # Blend images based on alpha channel
        blended = (1 - alpha_3d) * original + alpha_3d * overlay_rgb

        # Create output directory if it doesn't exist
        output_dir = "processed_images"
        os.makedirs(output_dir, exist_ok=True)

        # Generate output path
        filename = os.path.basename(original_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_overlayed{ext}")

        # Save the result
        cv2.imwrite(output_path, blended.astype(np.uint8))

        # Update result dictionary
        result['success'] = True
        result['output_path'] = output_path

    except Exception as e:
        result['error'] = str(e)

    return result

def process_with_overlay(original_path: str, overlay_path: str) -> Dict[str, Union[str, bool]]:
    """
    Process a single image pair with overlay

    Args:
        original_path (str): Path to the original image
        overlay_path (str): Path to the overlay PNG image

    Returns:
        dict: Dictionary containing:
            - success (bool): Whether processing was successful
            - output_path (str): Path to processed image if successful, None if failed
            - error (str): Error message if processing failed, None if successful
    """
    return overlay_images(original_path, overlay_path)

if __name__ == "__main__":
    # Example usage
    original_path = "path/to/original/image.jpg"
    overlay_path = "path/to/overlay/mask.png"
    result = process_with_overlay(original_path, overlay_path)

    if result['success']:
        print(f"Successfully processed image. Saved to: {result['output_path']}")
    else:
        print(f"Failed to process image: {result['error']}")