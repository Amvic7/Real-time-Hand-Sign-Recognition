import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

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

# Initialize a deque (sliding window) to keep track of recent predictions
prediction_window = deque(maxlen=10)  # Adjust size for more/less stability
stable_letter = None

# Start video capture
cap = cv2.VideoCapture(0)

# Variable to store the sentence
sentence = ""
last_prediction = None
min_stable_count = 5  # Number of consecutive predictions needed for stability

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

        # Add the current prediction to the deque
        prediction_window.append(predicted_sign)

        # Check if the same letter has been predicted consecutively enough times
        if prediction_window.count(predicted_sign) >= min_stable_count:
            stable_letter = predicted_sign

            # Append the stable letter to the sentence if it's different from the last one
            if stable_letter != last_prediction:
                last_prediction = stable_letter
                sentence += stable_letter
                print(f"Added to sentence: {stable_letter}")

        # Optional: Display the current stable prediction on the frame
        cv2.putText(frame, f'Stable Sign: {stable_letter}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        # No hand detected, add a space to the sentence
        if last_prediction != " ":
            sentence += " "
            last_prediction = " "
            print("No hand detected, adding space")

    # Display the video feed with the sentence
    cv2.putText(frame, f'Sentence: {sentence}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Add instruction for erasing the sentence
    cv2.putText(frame, "Press 'e' to erase the sentence", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the video feed
    cv2.imshow('Hand Sign Recognition', frame)

    # Check for keypresses
    key = cv2.waitKey(1) & 0xFF  # Wait for a key press event
    if key == ord('q'):  # Press 'q' to exit
        break
    elif key == ord('e'):  # Press 'e' to erase the sentence
        sentence = ""  # Clear the sentence
        last_prediction = None  # Reset last prediction
        print("Sentence erased")

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print the final sentence
print("Final Sentence:", sentence)
