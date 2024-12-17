import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/emotion_model_final.h5')

# Emotion labels from FER2013 dataset
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start the webcam
cap = cv2.VideoCapture(0)

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face from the grayscale frame
        face = gray[y:y+h, x:x+w]

        # Resize the face to 48x48 (model input size)
        face = cv2.resize(face, (48, 48))

        # Normalize pixel values to [0, 1]
        face = face / 255.0

        # Add channel dimension (for grayscale: (48, 48, 1))
        face = np.expand_dims(face, axis=-1)

        # Add batch dimension (1, 48, 48, 1)
        face = np.expand_dims(face, axis=0)

        # Predict emotion
        prediction = model.predict(face)
        emotion_index = np.argmax(prediction)
        emotion = emotions[emotion_index]

        # Display the emotion label on the frame
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Emotion Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
