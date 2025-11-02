# Drone Image Recognition with YOLO and Optimal Transport

An object recognition system for drone images combining **YOLOv11** for detection and **Optimal Transport theory** for classification. The model recognizes 6 classes: people, bike, vehicle, truck, bus, and motorcycle.

## 1. Architecture

**Pipeline**: Drone Image → YOLOv11 → Feature Extraction → Optimal Transport Distance → KNN Classifier

**Components**:
- **YOLOv11m**: Object detection model
- **ResNet50**: Feature extractor (optional, can use RGB directly)
- **GeomLoss**: Sinkhorn distance computation
- **KNN**: Classifier based on class prototypes

## 2. Installation

```bash
pip install torch torchvision ultralytics geomloss opencv-python scikit-learn joblib matplotlib tqdm
```

## 3. Usage

**Prediction**:
```bash
python predict.py
```

**Evaluation**:
```bash
python metrics.py
```

## 4. Configuration

Key parameters in `predict.py`:
```python
RGB = True          # Use RGB directly or ResNet features
K = 100            # Number of prototypes per class
n_neighbors = 10   # KNN neighbors
```

YOLO training: 100 epochs, batch size 32, image size 640x640

## 5. Optimal Transport Approach

Optimal Transport measures distances between probability distributions. Each image is a distribution in feature space, and each class has K prototype distributions. Sinkhorn distance computes the transformation cost between distributions, enabling robust classification via KNN.
