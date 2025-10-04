# 🧠 Deep Learning Projects Portfolio

A collection of **6 deep learning projects** covering computer vision, NLP, and audio processing with state-of-the-art neural network architectures.

---

## 📊 Projects Summary

| Project | Domain | Architecture | Accuracy | Dataset |
|---------|--------|--------------|----------|---------|
| [Facial Emotion Recognition](#-facial-emotion-recognition) | Computer Vision | CNN | 60-70% | FER2013 (35K images) |
| [Face Mask Detection](#-face-mask-detection) | Computer Vision | CNN | 85-90% | Mask images |
| [CIFAR-10 Classification](#-cifar-10-classification) | Computer Vision | ResNet50 Transfer Learning | 75-85% | 60K images |
| [Movie Sentiment Analysis](#-movie-sentiment-analysis) | NLP | LSTM | 85-90% | IMDB (50K reviews) |
| [Toxic Comment Classification](#-toxic-comment-classification) | NLP | Bidirectional LSTM | 85-90% | 160K comments |
| [Capuchin Bird Detection](#-capuchin-bird-detection) | Audio Processing | CNN on Spectrograms | >85% | Audio recordings |

---

## 😊 Facial Emotion Recognition

Classifies facial expressions into 7 emotions using CNN.

**Key Features:**
- 7 emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Grayscale images (48×48 pixels)
- 3 Conv blocks with MaxPooling and Dropout
- Adam optimizer with early stopping

**Tech Stack:** `TensorFlow` `Keras` `NumPy` `Scikit-learn`

**Applications:** Mental health monitoring, human-computer interaction, customer sentiment analysis

---

## 😷 Face Mask Detection

Binary classification system for automated mask compliance checking.

**Key Features:**
- Binary classification (with mask / without mask)
- RGB images resized to 128×128 pixels
- 3 Conv2D layers with progressive filters (32→64→128)
- Dropout regularization for robustness
- Automated image preprocessing pipeline

**Tech Stack:** `TensorFlow` `OpenCV` `PIL` `Scikit-learn`

**Applications:** Public health monitoring, access control systems, safety compliance

---

## 🖼️ CIFAR-10 Classification

Multi-class image classification using ResNet50 transfer learning.

**Key Features:**
- 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Transfer learning with pre-trained ResNet50 (ImageNet weights)
- Progressive upsampling strategy: 32×32 → 256×256
- Custom classification head with BatchNormalization
- Multi-environment support (Kaggle/Colab/Local)

**Tech Stack:** `TensorFlow` `Keras` `ResNet50` `NumPy`

**Applications:** Object recognition, content management, visual search engines

---

## 🎬 Movie Sentiment Analysis

Analyzes IMDB movie reviews using LSTM neural networks.

**Key Features:**
- Binary sentiment classification (positive/negative)
- 50,000 balanced movie reviews
- Sequential LSTM architecture for text processing
- Embedding layer (128 dimensions)
- Real-time prediction with confidence scores

**Tech Stack:** `TensorFlow` `Keras` `Pandas` `Matplotlib`

**Applications:** Entertainment industry analysis, marketing insights, recommendation systems

---

## 🛡️ Toxic Comment Classification

Multi-label content moderation system using bidirectional LSTM.

**Key Features:**
- 6 toxicity categories: toxic, severe_toxic, obscene, threat, insult, identity_hate
- Bidirectional LSTM for context understanding
- Interactive Gradio web interface
- Real-time toxicity scoring
- Multi-label classification capability

**Tech Stack:** `TensorFlow` `Keras` `Gradio` `Pandas`

**Applications:** Social media moderation, online forum safety, customer service monitoring

---

## 🦜 Capuchin Bird Detection

Audio-based bird call detection using CNN on spectrograms.

**Key Features:**
- Binary classification (Capuchin vs Non-Capuchin)
- Audio spectrogram analysis (1491×257×1)
- CNN architecture for feature extraction
- Processes 1 hour of audio in <5 minutes
- CSV export for scientific analysis

**Tech Stack:** `TensorFlow` `TensorFlow-IO` `NumPy` `Matplotlib`

**Applications:** Wildlife conservation, biodiversity monitoring, ecological research

---

## 🛠️ Technical Capabilities

**Deep Learning Architectures:**
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (LSTM, Bidirectional LSTM)
- Transfer Learning (ResNet50)
- Custom Neural Network Design

**Specialized Skills:**
- Computer Vision (image classification, object detection)
- Natural Language Processing (sentiment analysis, text classification)
- Audio Signal Processing (spectrogram analysis)
- Multi-label Classification
- Transfer Learning & Fine-tuning

**Tools & Frameworks:**
`TensorFlow` `Keras` `OpenCV` `PIL` `Gradio` `TensorFlow-IO` `Scikit-learn` `Pandas` `NumPy`

---

## 📂 Repository Structure

```
Deep_Learning-Projects/
├── Facial_Emotion_Recognition/     # CNN for emotion detection
├── Face_Mask_Detection/            # COVID safety compliance
├── CIFAR10_Classification/         # ResNet50 transfer learning
├── Movie_Sentiment_Analysis/       # LSTM for reviews
├── Toxic_Comment_Classification/   # BiLSTM content moderation
├── Capuchin_Bird_Detection/        # Audio CNN classification
└── README.md
```

---

## 🎯 Key Highlights

✅ **6 Production-Ready Projects** across multiple domains  
✅ **State-of-the-Art Architectures** (CNN, LSTM, Transfer Learning)  
✅ **Multiple Modalities** - Vision, Text, Audio  
✅ **Real-World Applications** with business impact  
✅ **High Accuracy Models** (80-90% across projects)  
✅ **Interactive Interfaces** (Gradio web apps)  

---

## 💡 Future Enhancements

- Implement Transformer models (BERT, ViT)
- Deploy models as REST APIs
- Add real-time webcam/microphone integration
- Build unified dashboard for all models
- Expand to video and multi-modal learning

---

## 📫 Contact

- **LinkedIn**: [Mohamed Makki](https://www.linkedin.com/in/mohamed-makki-ab5a10302/)
- **Email**: makki0749@gmail.com
- **Kaggle**: [Mohamed Makki](https://www.kaggle.com/mohamedmakkiabdelaal)

---

⭐ **If you find this portfolio helpful, please star this repository!**