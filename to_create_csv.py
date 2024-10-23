import cv2
import mediapipe as mp
import pandas as pd
import os

# Path to the folder where your images are stored
image_folder = "hand_sign_images"
csv_file_path = "hand_sign_data.csv"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# List to store extracted data
data = []

# Process each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Extract the sign label from the filename
        sign_label = filename.split('_')[0]  # Assumes the filename format is like 'A_1.jpg'
        
        # Read the image
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        
        # Convert the image to RGB and process it with MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark positions (x, y, z for each landmark)
                landmark_data = [landmark.x for landmark in hand_landmarks.landmark] + \
                                [landmark.y for landmark in hand_landmarks.landmark] + \
                                [landmark.z for landmark in hand_landmarks.landmark]
                
                # Append the label and landmarks to the data list
                data.append([sign_label] + landmark_data)

# Convert the data to a pandas DataFrame and save it as a CSV file
columns = ['sign'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)]
df = pd.DataFrame(data, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

print(f"CSV file created successfully at {csv_file_path}")
