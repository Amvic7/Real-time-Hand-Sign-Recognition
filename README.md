# Hand Sign Language to Text Recognition

This project enables real-time hand sign language recognition, converting recognized signs into corresponding text. It uses a trained neural network model to detect and classify hand signs from video input.

## Features
- Recognizes hand signs (A-Z) using a webcam
- Dynamically generates and displays a sentence based on recognized signs
- Adds a space when no hand is detected
- Allows erasing the entire sentence by pressing the `E` key

## How to Use
1. Run the Python script to start the webcam and hand sign recognition.
2. Show hand signs in front of the camera for real-time recognition.
3. The recognized signs will appear as text on the screen.
4. Press `E` to erase the current sentence.

## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- MediaPipe

## Setup
1. Clone this repository:
    ```bash
    git clone https://github.com/Amvic7/Real-time-Hand-Sign-Recognition.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the program:
    ```bash
    python sign_language_recognition.py
    ```

## Dataset
- The hand sign data is collected from custom images using MediaPipe and is trained using a neural network in Keras.

## Model
- The trained model (`hand_sign_model.keras`) is included for prediction purposes. The model can recognize hand signs from A-Z.

## Contributors
- [Amvic](https://github.com/Amvic7)
- [Atharv Dhupkar](https://github.com/atharv-dhupkar22)
- [Atharva Bhosale](https://github.com/AT8Cool)
- [Gauresh Bhoir](https://github.com/Slugget98)

## License
This project is licensed under the MIT License.
