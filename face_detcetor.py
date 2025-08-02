import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load Haar cascades for face detection
face_cascade = cv2.CascadeClassifier(r"haarcascades/haarcascade_frontalface_alt.xml")

if face_cascade.empty():
    print("Error loading face cascade!")
    exit()

# Load the trained emotion model
emotion_model = load_model("model.h5")

# Define your emotion labels (must match your training order)
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]  # Change if needed

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for detection (detection only)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.1, 5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract face ROI in COLOR (because model expects 3 channels)
        roi_color = frame[y:y + h, x:x + w]
        roi_color = cv2.resize(roi_color, (48, 48))

        # Preprocess the image for the model
        roi_color = roi_color.astype("float") / 255.0
        roi_color = img_to_array(roi_color)
        roi_color = np.expand_dims(roi_color, axis=0)

        # Predict emotion
        predictions = emotion_model.predict(roi_color)
        emotion_index = np.argmax(predictions[0])
        emotion = emotion_labels[emotion_index]
        confidence = predictions[0][emotion_index]

        # Display emotion with confidence
        cv2.putText(frame, f"{emotion} ({confidence*100:.1f}%)",
                    (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)

    # Show the output window
    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
