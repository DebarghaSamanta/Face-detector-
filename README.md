Real-Time Facial Emotion Recognition (FER)

This project is a Real-Time Facial Emotion Recognition System built using:
OpenCV (cv2) for face detection (Haar Cascades)
Custom CNN model (TensorFlow/Keras) for emotion classification

Webcam feed for live predictions
It detects faces and classifies emotions like Anger, Disgust, Fear, Happy, Neutral, Sadness, and Surprise in real time.

🧠 Model Details
I tested multiple architectures:
VGG16 Transfer Learning → Accuracy ~62%
Other Custom CNNs → Accuracy ~64-66%

FER_Model Highlights:
Input: 48×48×3 image
5 convolutional blocks (64 → 512 filters)
Batch Normalization + Dropout for regularization
Output: 7-class softmax

Webcam Feed:
Green rectangle shows detected face
Emotion label & confidence score displayed above
