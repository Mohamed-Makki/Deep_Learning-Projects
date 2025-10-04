# CIFAR-10 Classification: ResNet50 Transfer Learning

A deep learning project that classifies images into 10 categories using the CIFAR-10 dataset and ResNet50 transfer learning architecture.

## Overview

This project implements a multi-class image classification system using transfer learning with ResNet50. The model can accurately classify images into 10 distinct object categories from the famous CIFAR-10 dataset, demonstrating the power of pre-trained networks for computer vision tasks.

## Dataset

- **CIFAR-10 Dataset**: 60,000 32×32 color images
- **Classes**: 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training Images**: 50,000 images
- **Test Images**: 10,000 images
- **Image Size**: 32×32 pixels with RGB channels
- **Split**: 80% training, 20% testing

## Requirements

```
numpy
pandas
matplotlib
opencv-python
pillow
scikit-learn
tensorflow
zipfile
```

## Installation & Setup

### For Kaggle:
1. Go to Kaggle → Datasets → Search "CIFAR-10"
2. Add the CIFAR-10 dataset to your notebook
3. The dataset will be available at `/kaggle/input/cifar-10/`
4. Run the script directly - paths are auto-configured

### For Local Development:
1. Clone the repository
2. Install required packages:
```bash
pip install numpy pandas matplotlib opencv-python pillow scikit-learn tensorflow
```
3. Download CIFAR-10 dataset and place in project directory

### For Google Colab:
1. Upload CIFAR-10 ZIP file to Colab
2. The script will auto-detect and extract the data

## Dataset Structure

### Kaggle Dataset Path:
```
/kaggle/input/cifar-10/
├── train/
│   ├── 1.png
│   ├── 2.png
│   └── ... (50,000 images)
└── trainLabels.csv    # Contains id and label columns
```

### Required Files:
- **trainLabels.csv**: Contains image IDs and corresponding class labels
- **train/ folder**: Contains 50,000 PNG images named by ID (1.png, 2.png, etc.)

## Usage

### On Kaggle:
1. Add CIFAR-10 dataset to your notebook
2. Run the script directly:
```bash
python cifar10_classification.py
```
The script automatically detects Kaggle environment and sets correct paths.

### On Local Machine:
1. Update the ZIP path in the script:
```python
ZIP_PATH = r'C:\path\to\your\cifar-10.zip'
```
2. Run the script

The script will:
- Extract and load the CIFAR-10 dataset
- Preprocess images and encode labels
- Build a transfer learning model with ResNet50
- Train the model with validation monitoring
- Evaluate performance and visualize results
- Demonstrate classification on test samples

## Model Architecture

**Transfer Learning with ResNet50:**

1. **Input Processing**:
   - Input: 32×32×3 CIFAR-10 images
   - UpSampling layers: 32×32 → 64×64 → 128×128 → 256×256

2. **Pre-trained Base**:
   - ResNet50 (ImageNet weights, no top layers)
   - Frozen layers for feature extraction

3. **Custom Classification Head**:
   - Flatten layer
   - BatchNormalization
   - Dense(128) + ReLU + Dropout(0.5)
   - BatchNormalization  
   - Dense(64) + ReLU + Dropout(0.5)
   - BatchNormalization
   - Dense(10) + Softmax (10 classes)

## Results

- **Training Accuracy**: ~85-95%
- **Validation Accuracy**: ~80-90%
- **Test Accuracy**: ~75-85%
- **Model**: ResNet50 Transfer Learning with custom head

## Key Features

- ✅ **Transfer Learning**: Leverages pre-trained ResNet50 features
- ✅ **Upsampling Strategy**: Handles CIFAR-10's small image size
- ✅ **Batch Normalization**: Improved training stability
- ✅ **Dropout Regularization**: Prevents overfitting
- ✅ **Multi-Environment Support**: Works on Kaggle/Colab/Local
- ✅ **Comprehensive Visualization**: Training curves and metrics
- ✅ **Robust Error Handling**: Fallback to synthetic data

## CIFAR-10 Classes

| Index | Class Name | Description |
|-------|------------|-------------|
| 0 | Airplane | Aircraft/planes |
| 1 | Automobile | Cars/vehicles |
| 2 | Bird | Various bird species |
| 3 | Cat | Domestic cats |
| 4 | Deer | Wild deer |
| 5 | Dog | Domestic dogs |
| 6 | Frog | Amphibians |
| 7 | Horse | Horses |
| 8 | Ship | Maritime vessels |
| 9 | Truck | Trucks/lorries |

## Applications

- **Object Recognition Systems**: General-purpose image classification
- **Content Management**: Automated image categorization
- **Visual Search Engines**: Image-based search and retrieval
- **Educational Tools**: Computer vision learning platforms
- **Quality Control**: Automated product classification

## Technical Highlights

### Transfer Learning Benefits:
- **Faster Training**: Pre-trained features accelerate learning
- **Better Performance**: ImageNet knowledge transfers to CIFAR-10
- **Reduced Data Requirements**: Effective with smaller datasets
- **Feature Reusability**: Leverages established visual patterns

### Upsampling Strategy:
- **Problem**: ResNet50 requires 224×224+ inputs, CIFAR-10 is 32×32
- **Solution**: Progressive upsampling (32→64→128→256)
- **Advantage**: Maintains spatial relationships while scaling

## File Structure

```
├── CIFAR-10 Image Classification.هpynb    # Main project script
├── cifar-10.zip                		   # CIFAR-10 dataset (download separately)
├── cifar-10/                   		   # Extracted dataset directory
│   ├── train/                 			   # Training images
│   └── trainlabels.csv                    # Image labels
└──README.md                               # Project documentation
```

## Performance Optimization

- **Learning Rate**: Low learning rate (2e-5) for transfer learning
- **Batch Size**: 32 for stable training
- **Optimizer**: RMSprop for adaptive learning
- **Regularization**: Dropout + BatchNormalization
- **Validation**: 10% validation split for monitoring

## Usage Example

```python
# Classify a CIFAR-10 image
image = load_image('path/to/cifar10/image.png')  # 32x32x3
pred_class, class_name, confidence = predict_cifar10_class(model, image, labels_dict)
print(f"Prediction: {class_name} (Confidence: {confidence:.2f})")
```

## Conclusion

This project successfully demonstrates the effectiveness of transfer learning for multi-class image classification. The ResNet50-based model achieved excellent performance on the CIFAR-10 dataset, proving that pre-trained networks can be successfully adapted for specific classification tasks through proper architectural modifications.

The implementation showcases advanced deep learning techniques including transfer learning, progressive upsampling, and regularization strategies. This approach is particularly valuable for computer vision applications where leveraging pre-trained features can significantly improve performance while reducing training time and computational requirements.

The project serves as a comprehensive example of applying state-of-the-art computer vision techniques to real-world classification problems and demonstrates how transfer learning can bridge the gap between different domain requirements.
