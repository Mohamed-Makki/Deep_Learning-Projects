# Face Mask Detection: CNN Classification System

A deep learning project that detects face masks in images using Convolutional Neural Networks (CNN) to support public health safety measures.

## Overview

This project implements a computer vision system for automated face mask detection. The model analyzes facial images to classify whether a person is wearing a mask or not, supporting compliance monitoring during health emergencies.

## Dataset

- **Images**: Facial images in two categories (with_mask / without_mask)
- **Input Size**: Images resized to 128×128 pixels
- **Classes**: Binary classification (0 = Without Mask, 1 = With Mask)
- **Split**: 80% training, 20% testing
- **Format**: RGB color images (JPG/PNG)

## Requirements

```
numpy
pandas
matplotlib
opencv-python
pillow
scikit-learn
tensorflow
seaborn
```

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install numpy pandas matplotlib opencv-python pillow scikit-learn tensorflow seaborn
```
3. Organize your dataset in the following structure:
```
Datasets/
├── with_mask/
│   ├── with_mask_1.jpg
│   ├── with_mask_2.jpg
│   └── ...
└── without_mask/
    ├── without_mask_1.jpg
    ├── without_mask_2.jpg
    └── ...
```

## Usage

1. Update the data paths in the script:
```python
WITH_MASK_PATH = 'path/to/your/with_mask/folder'
WITHOUT_MASK_PATH = 'path/to/your/without_mask/folder'
```

2. Run the main script:
```bash
python mask_detection.ipynb
```

The script will:
- Load and preprocess image data
- Create binary labels for classification
- Build and train a CNN model
- Evaluate model performance
- Provide a prediction system for new images

## Model Architecture

**CNN Architecture:**
- **Conv2D Layer 1**: 32 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D**: 2×2 pool size
- **Conv2D Layer 2**: 64 filters, 3×3 kernel, ReLU activation  
- **MaxPooling2D**: 2×2 pool size
- **Conv2D Layer 3**: 128 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D**: 2×2 pool size
- **Flatten Layer**: Convert to 1D
- **Dense Layer 1**: 128 neurons, ReLU + Dropout(0.5)
- **Dense Layer 2**: 64 neurons, ReLU + Dropout(0.5)
- **Output Layer**: 2 neurons, Sigmoid activation

## Results

- **Training Accuracy**: ~90-95%
- **Validation Accuracy**: ~85-92%
- **Test Accuracy**: ~85-90%
- **Model**: CNN with dropout regularization

## Key Features

- ✅ Automated image preprocessing and resizing
- ✅ Robust CNN architecture with dropout layers
- ✅ Data normalization for optimal training
- ✅ Training progress visualization
- ✅ Comprehensive model evaluation
- ✅ Real-time prediction system
- ✅ Error handling for image processing

## Applications

- **Public Health Monitoring**: Automated mask compliance checking
- **Access Control Systems**: Entry monitoring for masked individuals
- **Retail & Transportation**: Safety compliance in public spaces
- **Healthcare Facilities**: Patient and visitor screening
- **Educational Institutions**: Campus safety monitoring

## Image Processing Pipeline

1. **Loading**: Read images from directories
2. **Resizing**: Standardize to 128×128 pixels
3. **Color Conversion**: Ensure RGB format
4. **Normalization**: Scale pixel values to [0,1] range
5. **Array Conversion**: Convert to NumPy arrays
6. **Batch Processing**: Organize for model training

## File Structure

```
├── Project/
│   ├── Face Mask Detection.ipynb         # Main project script
├── Datasets/                    		  # Dataset directory
│   ├── with_mask/          			  # Images with masks
│   └── without_mask/       			  # Images without masks
└── README.md              				  # Project documentation
```

## Usage Example

```python
# Predict mask status for a new image
image_path = "test_image.jpg"
prediction, status, confidence = predict_mask_status(model, image_path)
print(f"Result: {status} (Confidence: {confidence:.2f})")
```

## Conclusion

This project successfully demonstrates the power of Convolutional Neural Networks in solving real-world computer vision problems. The CNN model achieved excellent performance in detecting face masks, showcasing how deep learning can support public health initiatives through automated visual monitoring.

The implementation highlights essential computer vision techniques including image preprocessing, CNN architecture design, and model optimization. This approach proves invaluable for healthcare applications where automated compliance monitoring can enhance community safety and support human efforts in maintaining health protocols.

The project serves as a practical example of applying AI technology to address societal challenges and demonstrates how machine learning can contribute to public health and safety measures in critical situations.
