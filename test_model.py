import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the model
model = load_model('combined_aaa.keras')

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Define a function to preprocess the input frame
def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark positions
            landmark_data = [landmark.x for landmark in hand_landmarks.landmark] + \
                            [landmark.y for landmark in hand_landmarks.landmark] + \
                            [landmark.z for landmark in hand_landmarks.landmark]
            return np.array(landmark_data).reshape(1, -1)
    return None

# Start video capture
cap = cv2.VideoCapture(0)

# Variable to store the last predicted sign
last_prediction = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for the model
    data = preprocess_frame(frame)
    if data is not None:
        # Make a prediction
        prediction = model.predict(data)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class index
        predicted_sign = letters[predicted_class]  # Map index to letter

        # Display the predicted sign if it's different from the last one
        if predicted_sign != last_prediction:
            last_prediction = predicted_sign
            print(f"Predicted Sign: {predicted_sign}")  # Display the predicted sign

        # Optional: Display the predicted sign on the frame
        cv2.putText(frame, f'Predicted Sign: {predicted_sign}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow('Hand Sign Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
