import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Load the landmark data from the CSV file
data_df = pd.read_csv('combined_aaa.csv')

# Separate features and labels
X = data_df.drop(columns=['sign']).values  # Features (landmarks)
y = data_df['sign'].values  # Labels (letters)

# Encode labels to integers
label_mapping = {label: index for index, label in enumerate(np.unique(y))}
y_encoded = np.array([label_mapping[label] for label in y])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define the model architecture
model = models.Sequential()
model.add(layers.Input(shape=(X_train.shape[1],)))  # Input layer
model.add(layers.Dense(128, activation='relu'))  # Hidden layer
model.add(layers.Dense(64, activation='relu'))  # Hidden layer
model.add(layers.Dense(len(label_mapping), activation='softmax'))  # Output layer for classification

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the model in Keras format
model.save('combined_aaa.keras')
print("Model saved as 'combined_aaa.keras'!")
