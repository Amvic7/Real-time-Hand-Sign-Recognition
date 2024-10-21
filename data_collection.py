import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# Create a directory to save images
save_dir = "hand_sign_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create a list to store data
data = []

# Define the limit for data collection
samples_per_letter = 100  # Change this value to set the limit for each letter

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Ask for user input for the letter
    sign_label = input("Enter the letter to collect data (A-Z) or 'q' to quit: ").upper()

    if sign_label == 'Q':
        break

    if sign_label not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        print("Invalid input. Please enter a letter from A to Z.")
        continue

    print(f"Collecting data for letter: {sign_label}")
    collected_samples = 0

    while collected_samples < samples_per_letter:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB and process it
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            print("Hand landmarks detected.")
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark positions
                landmark_data = [landmark.x for landmark in hand_landmarks.landmark] + \
                                [landmark.y for landmark in hand_landmarks.landmark] + \
                                [landmark.z for landmark in hand_landmarks.landmark]

                # Append the sign label and landmark data to the list
                data.append([sign_label] + landmark_data)

                # Save the image with a filename indicating the letter and sample number
                img_filename = os.path.join(save_dir, f"{sign_label}_{collected_samples + 1}.jpg")
                cv2.imwrite(img_filename, frame)
                print(f"Saved image: {img_filename}")

                collected_samples += 1
                print(f"Collected {collected_samples}/{samples_per_letter} samples for letter: {sign_label}")

        else:
            print("No hand detected.")

        # Display the video feed
        cv2.imshow('Hand Sign Capture', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Convert collected data to a DataFrame and save to CSV
columns = ['sign'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)]
print("Collected data:", data)  # Debugging
df = pd.DataFrame(data, columns=columns)
print("DataFrame shape:", df.shape)  # Debugging
df.to_csv('hand_sign_data.csv', index=False)

print("Data collection completed!")
