import os
import time
import cv2
import numpy as np
import torch
import psutil
import mediapipe as mp
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import tensorflow as tf
from mtcnn import MTCNN
import onnxruntime as ort
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
warnings.filterwarnings('ignore')

class SystemMonitor:
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    @staticmethod
    def get_cpu_usage():
        """Get CPU usage percentage"""
        return psutil.cpu_percent(interval=1)

    @staticmethod
    def get_cpu_temperature():
        """Get CPU temperature (Raspberry Pi specific)"""
        try:
            temp = os.popen("vcgencmd measure_temp").readline()
            return float(temp.replace("temp=","").replace("'C\n",""))
        except:
            return None

class FDDBEvaluator:
    def __init__(self, dataset_path, annotation_file):
        self.dataset_path = dataset_path
        self.annotation_file = annotation_file

    def load_ground_truth(self):
        """Load ground truth annotations"""
        gt_data = []
        current_image = None
        current_boxes = []

        with open(self.annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    if current_image is not None:
                        full_image_path = os.path.join(self.dataset_path, current_image)
                        gt_data.append((full_image_path, np.array(current_boxes)))
                    current_image = line[2:]
                    current_boxes = []
                else:
                    try:
                        x1, y1, x2, y2 = map(float, line.split())
                        current_boxes.append([x1, y1, x2, y2])
                    except ValueError:
                        print(f"Warning: Could not parse line: {line}")
                        continue

        if current_image is not None:
            full_image_path = os.path.join(self.dataset_path, current_image)
            gt_data.append((full_image_path, np.array(current_boxes)))

        return gt_data

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def calculate_ap(self, predictions, ground_truth, iou_threshold=0.5):
        """Calculate Average Precision"""
        true_positives = []
        false_positives = []
        num_gt_faces = sum(len(gt) for _, gt in ground_truth)

        for (_, pred_boxes), (_, gt_boxes) in zip(predictions, ground_truth):
            if len(pred_boxes) == 0:
                continue

            matched = np.zeros(len(gt_boxes), dtype=bool)

            for pred_box in pred_boxes:
                max_iou = 0
                max_idx = -1

                for i, gt_box in enumerate(gt_boxes):
                    if not matched[i]:
                        iou = self.calculate_iou(pred_box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                            max_idx = i

                if max_iou >= iou_threshold and max_idx >= 0:
                    true_positives.append(1)
                    false_positives.append(0)
                    matched[max_idx] = True
                else:
                    true_positives.append(0)
                    false_positives.append(1)

        if not true_positives:
            return 0.0

        true_positives = np.cumsum(true_positives)
        false_positives = np.cumsum(false_positives)

        recall = true_positives / num_gt_faces
        precision = true_positives / (true_positives + false_positives)

        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11

        return ap

# Face Detection Models
class OpenCVHaarDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model_size = os.path.getsize(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') / (1024 * 1024)

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return [[x, y, x + w, y + h] for (x, y, w, h) in faces]

class MediaPipeDetector:
    def __init__(self):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=0
        )
        # Approximate size for MediaPipe face detection model
        self.model_size = 5  # MB (approximate)

    def detect(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        boxes = []
        if results.detections:
            ih, iw, _ = image.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                boxes.append([x, y, x + w, y + h])
        return boxes

class MTCNNDetector:
    def __init__(self):
        self.detector = MTCNN()
        self.model_size = 40  # MB (approximate)

    def detect(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(image_rgb)
        return [[int(box['box'][0]),
                 int(box['box'][1]),
                 int(box['box'][0] + box['box'][2]),
                 int(box['box'][1] + box['box'][3])] for box in detections]

class ShunyafaceDetector:
    def __init__(self):
        try:
            import shunyaface
            self.detector = shunyaface.FaceDetector()
            # Approximate size based on repository description
            self.model_size = 25  # MB (approximate)
        except ImportError:
            raise ImportError("Shunyaface not installed. Please install Shunyaface to use this detector.")

    def detect(self, image):
        """
        Detect faces using Shunyaface
        Args:
            image: numpy array of BGR image (OpenCV format)
        Returns:
            list of bounding boxes in format [x1, y1, x2, y2]
        """
        try:
            # Convert BGR to RGB as Shunyaface might expect RGB format
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Detect faces - assuming the API returns bounding boxes
            detections = self.detector.detect(image_rgb)
            # Convert detections to the standard format [x1, y1, x2, y2]
            boxes = []
            for det in detections:
                # Assuming Shunyaface returns [x, y, width, height]
                x, y, w, h = det
                boxes.append([int(x), int(y), int(x + w), int(y + h)])
            return boxes
        except Exception as e:
            print(f"Error in Shunyaface detection: {str(e)}")
            return []

class DlibHOGDetector:
    def __init__(self):
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            # Approximate size for Dlib HOG detector
            self.model_size = 15  # MB (approximate)
        except ImportError:
            raise ImportError("Dlib not installed. Please install dlib to use this detector.")

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dets = self.detector(gray, 1)
        return [[d.left(), d.top(), d.right(), d.bottom()] for d in dets]

class YOLOv8Detector:
    def __init__(self):
        print("Downloading YOLOv8 model...")
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection",
            filename="model.pt"
        )
        self.model = YOLO(model_path)
        self.model_size = os.path.getsize(model_path) / (1024 * 1024)

    def detect(self, image):
        results = self.model(image)
        boxes = []
        for r in results:
            for box in r.boxes.xyxy:
                boxes.append([
                    float(box[0]), float(box[1]),
                    float(box[2]), float(box[3])
                ])
        return boxes
class TFLiteDetector:
    def __init__(self):
        import tensorflow as tf

        # Download and load the model
        model_url = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
        model_path = tf.keras.utils.get_file('face_detection.tflite', model_url)

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        import os
        self.model_size = os.path.getsize(model_path) / (1024 * 1024)

    def detect(self, image):
        import tensorflow as tf
        import numpy as np

        # Preprocess image
        input_shape = self.input_details[0]['shape'][1:3]
        input_image = cv2.resize(image, input_shape)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = (input_image - 127.5) / 127.5
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype(np.float32)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()

        # Get detection results
        detection_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        detection_scores = self.interpreter.get_tensor(self.output_details[1]['index'])[0]

        # Filter and convert detections
        height, width = image.shape[:2]
        boxes = []

        for i in range(len(detection_scores)):
            if detection_scores[i] > 0.5:
                box = detection_boxes[i]
                # Convert normalized coordinates to pixel coordinates
                y_min, x_min, y_max, x_max = box
                x1 = int(max(0, x_min * width))
                y1 = int(max(0, y_min * height))
                x2 = int(min(width, x_max * width))
                y2 = int(min(height, y_max * height))
                boxes.append([x1, y1, x2, y2])

        return boxes

class BlazeFaceDetector:
    def __init__(self):
        import mediapipe as mp
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=1  # Use the full-range model
        )
        self.model_size = 3  # MB (approximate)

    def detect(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(image_rgb)
        boxes = []

        if results.detections:
            ih, iw, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = max(0, int(bbox.xmin * iw))
                y1 = max(0, int(bbox.ymin * ih))
                x2 = min(iw, int((bbox.xmin + bbox.width) * iw))
                y2 = min(ih, int((bbox.ymin + bbox.height) * ih))

                # Ensure valid box coordinates
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])

        return boxes

class OpenVINODetector:
    def __init__(self):
        from openvino.runtime import Core
        import requests
        import os

        # Download the face detection model
        model_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.xml"
        weights_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.bin"

        model_path = "face-detection-retail-0004.xml"
        weights_path = "face-detection-retail-0004.bin"

        # Download files if they don't exist
        if not os.path.exists(model_path):
            response = requests.get(model_url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
        if not os.path.exists(weights_path):
            response = requests.get(weights_url)
            with open(weights_path, 'wb') as f:
                f.write(response.content)

        # Initialize OpenVINO runtime and load model
        ie = Core()
        self.model = ie.read_model(model_path)
        self.compiled_model = ie.compile_model(self.model, "CPU")
        self.output_layer = self.compiled_model.output(0)

        self.model_size = (os.path.getsize(model_path) + os.path.getsize(weights_path)) / (1024 * 1024)

    def detect(self, image):
        import numpy as np

        # Preprocess image
        height, width = image.shape[:2]
        input_image = cv2.resize(image, (300, 300))
        input_image = input_image.transpose((2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)

        # Run inference
        results = self.compiled_model([input_image])[self.output_layer]

        # Process detections
        boxes = []
        for detection in results[0][0]:
            confidence = float(detection[2])
            if confidence > 0.5:
                x1 = int(max(0, detection[3] * width))
                y1 = int(max(0, detection[4] * height))
                x2 = int(min(width, detection[5] * width))
                y2 = int(min(height, detection[6] * height))

                # Ensure valid box coordinates
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])

        return boxes

def create_performance_visualizations(results):
    """
    Create comprehensive visualizations for face detection model comparisons
    """
    # Filter out models with zero or invalid performance
    results = {k: v for k, v in results.items()
               if v['average_precision'] > 0 and v['average_inference_time'] > 0}

    # Set style
    plt.style.use('default')

    # Create figure
    fig = plt.figure(figsize=(20, 15))

    # Get models list and color palette
    models = list(results.keys())
    palette = plt.cm.tab10(np.linspace(0, 1, len(models)))

    # 1. Bar plot for Average Precision
    plt.subplot(2, 2, 1)
    ap_scores = [results[model]['average_precision'] for model in models]

    bars = plt.bar(models, ap_scores, color=palette)
    plt.title('Average Precision by Model', fontsize=12, pad=20)
    plt.xlabel('Model', fontsize=10)
    plt.ylabel('Average Precision', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')

    # 2. Bar plot for Inference Time
    plt.subplot(2, 2, 2)
    inference_times = [results[model]['average_inference_time'] for model in models]

    bars = plt.bar(models, inference_times, color=palette)
    plt.title('Average Inference Time by Model', fontsize=12, pad=20)
    plt.xlabel('Model', fontsize=10)
    plt.ylabel('Time (seconds)', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}s',
                 ha='center', va='bottom')

    # 3. Memory Usage
    plt.subplot(2, 2, 3)
    memory_usage = [results[model]['max_memory_usage_mb'] for model in models]

    bars = plt.bar(models, memory_usage, color=palette)
    plt.title('Maximum Memory Usage by Model', fontsize=12, pad=20)
    plt.xlabel('Model', fontsize=10)
    plt.ylabel('Memory Usage (MB)', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}MB',
                 ha='center', va='bottom')

    # 4. Radar Plot for normalized metrics
    ax = plt.subplot(2, 2, 4, projection='polar')

    metrics = ['AP', 'Speed', 'Memory Efficiency', 'CPU Efficiency']
    num_metrics = len(metrics)
    angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
    angles += angles[:1]

    # Normalize and plot each model
    for idx, model in enumerate(models):
        values = []
        # Average Precision (higher is better)
        values.append(results[model]['average_precision'])
        # Speed (inverse of inference time, higher is better)
        values.append(1 / (results[model]['average_inference_time'] + 1e-10))
        # Memory efficiency (inverse of memory usage, higher is better)
        values.append(1 / (max(results[model]['max_memory_usage_mb'], 0.1)))
        # CPU efficiency (inverse of CPU usage, higher is better)
        cpu_usage = abs(results[model]['average_cpu_usage_percent'])
        values.append(1 / (cpu_usage + 1e-10))

        # Normalize values to 0-1 scale
        min_val = min(values)
        max_val = max(values)
        if max_val > min_val:
            values = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            values = [0.5 for _ in values]
        values += values[:1]

        ax.plot(angles, values, linewidth=1, label=model, color=palette[idx])
        ax.fill(angles, values, alpha=0.1, color=palette[idx])

    plt.title('Normalized Performance Metrics', fontsize=12, pad=20)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    # Position legend outside the plot
    legend = ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))

    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Save figure
    plt.savefig('model_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_performance_tradeoff(results):
    """
    Create scatter plot showing speed-accuracy tradeoff
    """
    # Filter out models with zero or invalid performance
    results = {k: v for k, v in results.items()
               if v['average_precision'] > 0 and v['average_inference_time'] > 0}

    plt.figure(figsize=(12, 8))

    models = list(results.keys())
    ap_scores = [results[model]['average_precision'] for model in models]
    inference_times = [results[model]['average_inference_time'] for model in models]
    memory_sizes = [results[model]['max_memory_usage_mb'] for model in models]

    # Get color palette
    palette = plt.cm.tab10(np.linspace(0, 1, len(models)))

    # Normalize bubble sizes
    min_size = 100
    max_size = 500
    sizes = np.array(memory_sizes)
    if sizes.max() > sizes.min():
        sizes = min_size + (max_size - min_size) * (sizes - sizes.min()) / (sizes.max() - sizes.min())
    else:
        sizes = np.full_like(sizes, (min_size + max_size) / 2)

    # Create scatter plot
    scatter = plt.scatter(inference_times, ap_scores, s=sizes,
                          alpha=0.6, c=palette)

    # Add labels for each point
    for i, model in enumerate(models):
        plt.annotate(model,
                     (inference_times[i], ap_scores[i]),
                     xytext=(10, 5),
                     textcoords='offset points',
                     fontsize=10,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.title('Speed-Accuracy Tradeoff', fontsize=14, pad=20)
    plt.xlabel('Inference Time (seconds)', fontsize=12)
    plt.ylabel('Average Precision', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Add size legend
    memory_ranges = [min(memory_sizes), np.mean(memory_sizes), max(memory_sizes)]
    legend_sizes = [min_size, (min_size + max_size)/2, max_size]
    legend_elements = [plt.scatter([], [], s=size, c='gray', alpha=0.6,
                                   label=f'{mem:.1f}MB')
                       for size, mem in zip(legend_sizes, memory_ranges)]

    plt.legend(handles=legend_elements,
               title='Memory Usage',
               title_fontsize=10,
               fontsize=9,
               bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.savefig('speed_accuracy_tradeoff.png', bbox_inches='tight', dpi=300)
    plt.close()

def evaluate_detector(detector, evaluator, gt_data, subset_size=None):
    """Enhanced evaluation function with system metrics"""
    predictions = []
    metrics = {
        "total_time": 0,
        "max_memory": 0,
        "total_cpu_usage": 0,
        "max_temperature": 0,
        "model_size_mb": getattr(detector, 'model_size', 0)
    }

    if subset_size is not None:
        gt_data = gt_data[:subset_size]

    num_images = len(gt_data)
    system_monitor = SystemMonitor()

    print(f"\nProcessing {num_images} images...")

    for i, (image_path, _) in enumerate(gt_data, 1):
        try:
            # Record starting metrics
            start_memory = system_monitor.get_memory_usage()
            start_cpu = system_monitor.get_cpu_usage()
            temperature = system_monitor.get_cpu_temperature()

            # Process image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            # Perform detection and measure time
            start_time = time.time()
            boxes = detector.detect(image)
            inference_time = time.time() - start_time

            # Print progress
            if i % 10 == 0 or i == num_images:  # Update every 10 images or on the last image
                print(f"Progress: {i}/{num_images} images processed | "
                      f"Inference time: {inference_time:.3f}s | "
                      f"Memory delta: {system_monitor.get_memory_usage() - start_memory:.1f}MB | "
                      f"CPU usage: {system_monitor.get_cpu_usage() - start_cpu:.1f}%")

            # Update metrics
            metrics["total_time"] += inference_time
            metrics["max_memory"] = max(metrics["max_memory"],
                                        system_monitor.get_memory_usage() - start_memory)
            metrics["total_cpu_usage"] += system_monitor.get_cpu_usage() - start_cpu
            if temperature:
                metrics["max_temperature"] = max(metrics["max_temperature"], temperature)

            predictions.append((image_path, boxes))

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    # Calculate final metrics
    ap = evaluator.calculate_ap(predictions, gt_data)
    num_processed = len(predictions)

    return {
        "average_precision": ap,
        "average_inference_time": metrics["total_time"] / num_processed,
        "max_memory_usage_mb": metrics["max_memory"],
        "average_cpu_usage_percent": metrics["total_cpu_usage"] / num_processed,
        "max_temperature_celsius": metrics["max_temperature"] if metrics["max_temperature"] > 0 else None,
        "model_size_mb": metrics["model_size_mb"]
    }

def save_summary_to_file(results, filename='face_detection_summary.txt'):
    """Save evaluation results to a text file"""
    with open(filename, 'w') as f:
        f.write("Face Detection Models Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")

        # Detailed results for each model
        f.write("Detailed Results by Model\n")
        f.write("-" * 50 + "\n")
        for name, metrics in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Average Precision: {metrics['average_precision']:.4f}\n")
            f.write(f"  Inference Time: {metrics['average_inference_time']:.4f} seconds\n")
            f.write(f"  Max Memory Usage: {metrics['max_memory_usage_mb']:.2f} MB\n")
            f.write(f"  Average CPU Usage: {metrics['average_cpu_usage_percent']:.2f}%\n")
            if metrics['max_temperature_celsius']:
                f.write(f"  Max CPU Temperature: {metrics['max_temperature_celsius']:.1f}°C\n")
            f.write(f"  Model Size: {metrics['model_size_mb']:.2f} MB\n")

        # Best models by category
        if results:
            f.write("\nBest Models by Category\n")
            f.write("-" * 50 + "\n")

            best_accuracy = max(results.items(), key=lambda x: x[1]['average_precision'])
            best_speed = min(results.items(), key=lambda x: x[1]['average_inference_time'])
            best_memory = min(results.items(), key=lambda x: x[1]['max_memory_usage_mb'])

            f.write(f"Best Accuracy: {best_accuracy[0]} (AP: {best_accuracy[1]['average_precision']:.4f})\n")
            f.write(f"Best Speed: {best_speed[0]} (Inference Time: {best_speed[1]['average_inference_time']:.4f}s)\n")
            f.write(f"Most Memory Efficient: {best_memory[0]} (Max Memory: {best_memory[1]['max_memory_usage_mb']:.2f}MB)\n")

            # Recommendations
            f.write("\nRecommendations by Use Case\n")
            f.write("-" * 50 + "\n")

            f.write("High Accuracy Applications (e.g., Security Systems):\n")
            f.write(f"  Recommended: {best_accuracy[0]}\n")

            fast_and_efficient = min(results.items(),
                                     key=lambda x: x[1]['average_inference_time'] * x[1]['max_memory_usage_mb'])
            f.write("\nReal-time Applications (e.g., Video Conferencing):\n")
            f.write(f"  Recommended: {fast_and_efficient[0]}\n")

            edge_score = min(results.items(),
                             key=lambda x: (x[1]['model_size_mb'] * x[1]['max_memory_usage_mb'] *
                                            x[1]['average_inference_time']))
            f.write("\nEdge Devices (e.g., Mobile, Raspberry Pi):\n")
            f.write(f"  Recommended: {edge_score[0]}\n")


def main():
    # Initialize paths
    dataset_path = "/home/eugene/Downloads/Dataset_FDDB/images"  # Update this path
    annotation_file = "/home/eugene/Downloads/Dataset_FDDB/label.txt"
    subset_size = int(input("Enter the number of images to test (0 for all images): ") or "0")

    if subset_size == 0:
        subset_size = None

    # Initialize evaluator
    evaluator = FDDBEvaluator(dataset_path, annotation_file)

    # Load ground truth data
    print("\nLoading ground truth data...")
    gt_data = evaluator.load_ground_truth()
    print(f"Loaded {len(gt_data)} images with annotations")

    # Initialize all detectors with error handling
    detector_classes = {
        "OpenCV Haar": OpenCVHaarDetector,
        "MediaPipe": MediaPipeDetector,
        "MTCNN": MTCNNDetector,
        "Dlib HOG": DlibHOGDetector,
        "YOLOv8": YOLOv8Detector,
        "Shunyaface": ShunyafaceDetector,
        # "TFLite": TFLiteDetector,
        # "BlazeFace": BlazeFaceDetector,
        "OpenVINO": OpenVINODetector# Add the new detector
    }

    detectors = {}
    for name, detector_class in detector_classes.items():
        try:
            print(f"\nInitializing {name} detector...")
            detectors[name] = detector_class()
            print(f"{name} detector initialized successfully")
        except Exception as e:
            print(f"Error initializing {name} detector: {str(e)}")
            print(f"Skipping {name} detector")

    # Evaluate each detector
    results = {}
    for name, detector in detectors.items():
        print(f"\nEvaluating {name}...")
        try:
            metrics = evaluate_detector(detector, evaluator, gt_data, subset_size)
            results[name] = metrics

            print(f"\n{name} Results:")
            print(f"Average Precision: {metrics['average_precision']:.4f}")
            print(f"Average Inference Time: {metrics['average_inference_time']:.4f} seconds")
            print(f"Max Memory Usage: {metrics['max_memory_usage_mb']:.2f} MB")
            print(f"Average CPU Usage: {metrics['average_cpu_usage_percent']:.2f}%")
            if metrics['max_temperature_celsius']:
                print(f"Max CPU Temperature: {metrics['max_temperature_celsius']:.1f}°C")
            print(f"Model Size: {metrics['model_size_mb']:.2f} MB")

        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            continue

    # Print and save final summary
    print("\nGenerating and saving summary...")

    save_summary_to_file(results)
    print("Summary has been saved to 'face_detection_summary.txt'")

    # Generate visualizations
    print("\nGenerating performance visualizations...")
    try:
        create_performance_visualizations(results)
        print("Generated comprehensive performance comparison plots (saved as 'model_comparison.png')")

        plot_performance_tradeoff(results)
        print("Generated speed-accuracy tradeoff plot (saved as 'speed_accuracy_tradeoff.png')")
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")

    # Print final summary
    print("\nFinal Summary:")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Average Precision: {metrics['average_precision']:.4f}")
        print(f"  Inference Time: {metrics['average_inference_time']:.4f} seconds")
        print(f"  Max Memory Usage: {metrics['max_memory_usage_mb']:.2f} MB")
        print(f"  Average CPU Usage: {metrics['average_cpu_usage_percent']:.2f}%")
        if metrics['max_temperature_celsius']:
            print(f"  Max CPU Temperature: {metrics['max_temperature_celsius']:.1f}°C")
        print(f"  Model Size: {metrics['model_size_mb']:.2f} MB")

    # Find the best model for different criteria
    if results:
        best_accuracy = max(results.items(), key=lambda x: x[1]['average_precision'])
        best_speed = min(results.items(), key=lambda x: x[1]['average_inference_time'])
        best_memory = min(results.items(), key=lambda x: x[1]['max_memory_usage_mb'])

        print("\nBest Models by Category:")
        print("-" * 50)
        print(f"Best Accuracy: {best_accuracy[0]} (AP: {best_accuracy[1]['average_precision']:.4f})")
        print(f"Best Speed: {best_speed[0]} (Inference Time: {best_speed[1]['average_inference_time']:.4f}s)")
        print(f"Most Memory Efficient: {best_memory[0]} (Max Memory: {best_memory[1]['max_memory_usage_mb']:.2f}MB)")

        # Generate recommendations based on different use cases
        print("\nRecommendations by Use Case:")
        print("-" * 50)
        print("High Accuracy Applications (e.g., Security Systems):")
        print(f"  Recommended: {best_accuracy[0]}")
        print("\nReal-time Applications (e.g., Video Conferencing):")
        fast_and_efficient = min(results.items(),
                                 key=lambda x: x[1]['average_inference_time'] * x[1]['max_memory_usage_mb'])
        print(f"  Recommended: {fast_and_efficient[0]}")

        print("\nEdge Devices (e.g., Mobile, Raspberry Pi):")
        edge_score = min(results.items(),
                         key=lambda x: (x[1]['model_size_mb'] * x[1]['max_memory_usage_mb'] *
                                        x[1]['average_inference_time']))
        print(f"  Recommended: {edge_score[0]}")

if __name__ == "__main__":
    main()