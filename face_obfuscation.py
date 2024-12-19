# import cv2
# import numpy as np
# from mtcnn import MTCNN
# import os
# from typing import Dict, Union
# import logging
#
# # Set up logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)
#
# class MTCNNDetector:
#     def __init__(self):
#         try:
#             self.detector = MTCNN()
#             logger.debug("MTCNN detector initialized successfully")
#         except Exception as e:
#             logger.error(f"Failed to initialize MTCNN detector: {str(e)}")
#             raise
#
#     def detect_and_cover(self, image_path: str) -> Dict[str, Union[str, bool]]:
#         """
#         Detect faces in the image and cover them with black boxes
#
#         Args:
#             image_path (str): Path to input image
#
#         Returns:
#             dict: Dictionary containing:
#                 - success (bool): Whether processing was successful
#                 - output_path (str): Path to processed image if successful, None if failed
#                 - error (str): Error message if processing failed, None if successful
#         """
#         result = {
#             'success': False,
#             'output_path': None,
#             'error': None
#         }
#
#         try:
#             logger.debug(f"Processing image: {image_path}")
#
#             # Verify file exists
#             if not os.path.exists(image_path):
#                 raise FileNotFoundError(f"Image file not found: {image_path}")
#
#             # Read the image
#             image = cv2.imread(image_path)
#             if image is None:
#                 raise ValueError(f"Could not load image from {image_path}")
#
#             logger.debug(f"Image shape: {image.shape}")
#
#             # Convert to RGB for MTCNN
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#             # Detect faces
#             logger.debug("Detecting faces...")
#             detections = self.detector.detect_faces(image_rgb)
#             logger.debug(f"Found {len(detections)} faces")
#
#             # Draw black boxes over detected faces
#             for i, detection in enumerate(detections):
#                 # Get coordinates
#                 x, y, width, height = detection['box']
#                 logger.debug(f"Face {i+1} coordinates: x={x}, y={y}, width={width}, height={height}")
#
#                 # Ensure coordinates are valid
#                 x = max(0, x)
#                 y = max(0, y)
#                 width = min(width, image.shape[1] - x)
#                 height = min(height, image.shape[0] - y)
#
#                 # Draw black rectangle
#                 cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 0), -1)
#
#             # Create output directory if it doesn't exist
#             output_dir = "processed_images"
#             os.makedirs(output_dir, exist_ok=True)
#             logger.debug(f"Output directory ensured: {output_dir}")
#
#             # Generate output path
#             filename = os.path.basename(image_path)
#             name, ext = os.path.splitext(filename)
#             output_path = os.path.join(output_dir, f"{name}_processed{ext}")
#
#             # Save the result
#             success = cv2.imwrite(output_path, image)
#             if not success:
#                 raise IOError(f"Failed to save processed image to {output_path}")
#
#             logger.debug(f"Successfully saved processed image to: {output_path}")
#
#             # Update result dictionary
#             result['success'] = True
#             result['output_path'] = output_path
#
#         except Exception as e:
#             logger.error(f"Error processing image: {str(e)}")
#             result['error'] = str(e)
#
#         logger.debug(f"Returning result: {result}")
#         return result
#
# def process_image(image_path: str) -> Dict[str, Union[str, bool]]:
#     """
#     Process a single image with face detection and covering
#
#     Args:
#         image_path (str): Path to the image to process
#
#     Returns:
#         dict: Dictionary containing:
#             - success (bool): Whether processing was successful
#             - output_path (str): Path to processed image if successful, None if failed
#             - error (str): Error message if processing failed, None if successful
#     """
#     try:
#         detector = MTCNNDetector()
#         return detector.detect_and_cover(image_path)
#     except Exception as e:
#         logger.error(f"Error in process_image: {str(e)}")
#         return {
#             'success': False,
#             'output_path': None,
#             'error': str(e)
#         }
#
# if __name__ == "__main__":
#     # Example usage
#     image_path = "/path/to/your/image.jpg"
#     result = process_image(image_path)
#
#     if result['success']:
#         print(f"Successfully processed image. Saved to: {result['output_path']}")
#     else:
#         print(f"Failed to process image: {result['error']}")

from openvino.runtime import Core
import cv2
import numpy as np
import os
import logging
import requests
from typing import Dict, Union

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class OpenVINODetector:
    def __init__(self):
        """Initialize OpenVINO detector with face detection model"""
        try:
            # Initialize OpenVINO runtime
            self.ie = Core()
            logger.debug("OpenVINO Core initialized successfully")

            # Model URLs
            model_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.xml"
            weights_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.bin"

            # Local paths
            self.model_path = "face-detection-retail-0004.xml"
            self.weights_path = "face-detection-retail-0004.bin"

            # Download model files if they don't exist
            self._download_model_files(model_url, weights_url)

            # Load and compile model
            self.model = self.ie.read_model(self.model_path)
            self.compiled_model = self.ie.compile_model(self.model, "CPU")
            self.output_layer = self.compiled_model.output(0)

            logger.debug("OpenVINO model loaded and compiled successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OpenVINO detector: {str(e)}")
            raise

    def _download_model_files(self, model_url: str, weights_url: str):
        """Download model files if they don't exist"""
        try:
            if not os.path.exists(self.model_path):
                logger.debug(f"Downloading model from {model_url}")
                response = requests.get(model_url)
                with open(self.model_path, 'wb') as f:
                    f.write(response.content)

            if not os.path.exists(self.weights_path):
                logger.debug(f"Downloading weights from {weights_url}")
                response = requests.get(weights_url)
                with open(self.weights_path, 'wb') as f:
                    f.write(response.content)

            logger.debug("Model files downloaded successfully")

        except Exception as e:
            logger.error(f"Failed to download model files: {str(e)}")
            raise

    def detect_and_cover(self, image_path: str) -> Dict[str, Union[str, bool]]:
        """
        Detect faces in the image and cover them with black boxes

        Args:
            image_path (str): Path to input image

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
            logger.debug(f"Processing image: {image_path}")

            # Verify file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")

            logger.debug(f"Image shape: {image.shape}")

            # Preprocess image
            height, width = image.shape[:2]
            input_image = cv2.resize(image, (300, 300))
            input_image = input_image.transpose((2, 0, 1))
            input_image = np.expand_dims(input_image, axis=0)

            # Run inference
            logger.debug("Running face detection inference...")
            results = self.compiled_model([input_image])[self.output_layer]

            # Process detections and draw black boxes
            logger.debug("Processing detections...")
            for detection in results[0][0]:
                confidence = float(detection[2])
                if confidence > 0.5:
                    x1 = int(max(0, detection[3] * width))
                    y1 = int(max(0, detection[4] * height))
                    x2 = int(min(width, detection[5] * width))
                    y2 = int(min(height, detection[6] * height))

                    # Ensure valid box coordinates
                    if x2 > x1 and y2 > y1:
                        # Draw black rectangle
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
                        logger.debug(f"Drew black box at coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # Create output directory if it doesn't exist
            output_dir = "processed_images"
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Output directory ensured: {output_dir}")

            # Generate output path
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_processed{ext}")

            # Save the result
            success = cv2.imwrite(output_path, image)
            if not success:
                raise IOError(f"Failed to save processed image to {output_path}")

            logger.debug(f"Successfully saved processed image to: {output_path}")

            # Update result dictionary
            result['success'] = True
            result['output_path'] = output_path

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            result['error'] = str(e)

        logger.debug(f"Returning result: {result}")
        return result

def process_image(image_path: str) -> Dict[str, Union[str, bool]]:
    """
    Process a single image with face detection and covering

    Args:
        image_path (str): Path to the image to process

    Returns:
        dict: Dictionary containing:
            - success (bool): Whether processing was successful
            - output_path (str): Path to processed image if successful, None if failed
            - error (str): Error message if processing failed, None if successful
    """
    try:
        detector = OpenVINODetector()
        return detector.detect_and_cover(image_path)
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        return {
            'success': False,
            'output_path': None,
            'error': str(e)
        }

if __name__ == "__main__":
    # Example usage
    image_path = "/path/to/your/image.jpg"
    result = process_image(image_path)

    if result['success']:
        print(f"Successfully processed image. Saved to: {result['output_path']}")
    else:
        print(f"Failed to process image: {result['error']}")