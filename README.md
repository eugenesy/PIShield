# Pi SHIELD: Edge-Based Privacy-Preserving Framework for Object Detection

This project was developed as our final project for the Mobile Pervasive Intelligence Class (CSIE5411) at National Taiwan University.

Pi SHIELD is a privacy-preserving framework that leverages edge computing for secure object detection. The system performs face anonymization on edge devices (Raspberry Pi) before utilizing cloud resources for comprehensive object detection, ensuring privacy while maintaining functionality.

## 🌟 Key Features

- Real-time face detection and anonymization at the edge
- Privacy-first approach using hybrid edge-cloud architecture
- Comprehensive object detection using YOLOv8
- Web-based interface built with Streamlit
- Resource-efficient implementation optimized for Raspberry Pi

## 🏗️ Architecture

The system employs a privacy-first architecture:

1. **Frontend**: Streamlit-based web application for image capture and visualization
2. **Edge Device**: Raspberry Pi performing critical privacy-preserving operations (face detection and obfuscation)
3. **Server**: Any backend server for additional object detection tasks (can be self-hosted or cloud-based)

The core privacy protection happens at the edge device, ensuring sensitive data is anonymized before leaving the local environment. The server component is flexible and can be deployed anywhere according to your requirements.

## 🛠️ Technical Details

### Face Detection Model Performance

| Model | AP | Time(s) | Memory | Size |
|-------|-----|---------|---------|------|
| OpenCV | 0.557 | 0.324 | 13.1MB | 0.9MB |
| MTCNN | 0.859 | 0.590 | 145.5MB | 40.0MB |
| YOLOv8 | 0.805 | 0.630 | 66.0MB | 6.0MB |
| OpenVINO | 0.784 | 0.075 | 6.0MB | 2.4MB |

### System Requirements

- Raspberry Pi 4 (recommended) or newer
- Python 3.8+
- OpenVINO Toolkit
- YOLOv8
- Streamlit

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pi-shield.git
cd pi-shield
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up OpenVINO:
```bash
pip install openvino-dev
```

## 🚀 Usage

1. Start the Streamlit web interface:
```bash
streamlit run website.py
```

2. Access the interface through your browser at `http://localhost:8501`

3. Follow the on-screen instructions to capture and process images

## 🔒 Privacy Features

- Face detection and anonymization performed locally on edge device before any transmission
- No raw biometric data ever leaves the edge device
- Flexible server deployment options (self-hosted, private cloud, etc.)
- Complete control over data flow and processing
- Server receives only pre-anonymized images
- Privacy-preserving object detection pipeline

## 🎯 Applications

- Smart Cities: Traffic monitoring with privacy protection
- Retail Analytics: Anonymous customer behavior analysis
- Security Systems: Privacy-compliant surveillance
- Smart Home: Secure monitoring solutions

## 🛣️ Roadmap

- [ ] Extended anonymization capabilities for additional privacy-sensitive objects
- [ ] Enhanced edge-cloud communication security
- [ ] Real-time video processing support
- [ ] Mobile device support
- [ ] Additional object detection models

## 📚 Course Information

**Course**: Mobile Pervasive Intelligence (CSIE5411)  
**Institution**: National Taiwan University  
**Semester**: Fall 2024  
**Instructor**: Prof. Fang-Jing Wu

**Team Members**:
- Egon von Brüning
- Vincent Lin
- Eugene Sy

## 📚 Dataset

This project uses the Face Detection Dataset and Benchmark (FDDB) for evaluating our face detection models. FDDB is a standard benchmark dataset containing annotated face regions in a set of images.

If you use the FDDB dataset in your work, please cite:
```bibtex
@TechReport{fddbTech,
  author = {Vidit Jain and Erik Learned-Miller},
  title =  {FDDB: A Benchmark for Face Detection in Unconstrained Settings},
  institution =  {University of Massachusetts, Amherst},
  year = {2010},
  number = {UM-CS-2010-009}
}
```

The FDDB dataset includes:
- 5,171 face annotations in unconstrained settings
- Multiple pose angles and out-of-focus faces
- Varying image resolutions (363x450 to 229x410 pixels)
- Diverse real-world scenarios

## 🙏 Acknowledgments

- Face Detection Dataset and Benchmark (FDDB) for evaluation data
- OpenVINO team for their efficient inference framework
- YOLOv8 developers for object detection capabilities
