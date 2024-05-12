# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 01:41:36 2024
 
@author: Siva Shankar
"""

import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
original_dataset = pd.read_csv("E:/IEI/Docs/Crop_recommendation.csv")

# Separate features (X) and target (y)
X = original_dataset[['N', 'P', 'K', 'temperature', 'humidity', 'ph']]
y = original_dataset['label']

# Encode the target variable (crop labels) into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=22)
rf_classifier.fit(X_train, y_train)

# Train a Simple RNN model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units=256, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(np.expand_dims(X_train, axis=2), y_train, epochs=50, batch_size=32, validation_split=0.01)

# Make predictions using both models
rf_predictions = rf_classifier.predict(X_test)
rnn_predictions = np.argmax(model.predict(np.expand_dims(X_test, axis=2)), axis=1)

# Combine predictions using a voting classifier or any other method
combined_predictions = (rf_predictions + rnn_predictions) / 2

# Predict accuracy for the Random Forest m  odel
rf_accuracy = rf_classifier.score(X_test, y_test)

# Predict accuracy for the Simple RNN model
rnn_accuracy = model.evaluate(np.expand_dims(X_test, axis=2), y_test)[1] * 100

print("\nRandom Forest Model Accuracy:", rf_accuracy * 100)
print("\n\nSimple RNN Model Accuracy:", rnn_accuracy)

# Evaluate the combined predictions
accuracy = np.sum(combined_predictions == y_test) / len(y_test)
print("\n\nCombined Model Accuracy:", accuracy*100)

'''     MODEL SAVING     '''
# Save the Random Forest classifier
rf_model_path = "C:/Users/Siva Shankar/Desktop/random_forest_model.pkl"
joblib.dump(rf_classifier, rf_model_path)
print("Random Forest Model saved successfully at:", rf_model_path)

# Save the Simple RNN model
rnn_model_path = "C:/Users/Siva Shankar/Desktop/simple_rnn_model.h5"  # Specify the .h5 extension for Keras format
model.save(rnn_model_path)
print("Simple RNN Model saved successfully at:", rnn_model_path)

'''    TESTING PURPOSE '''
testing_data = pd.DataFrame({
    'N': [4],
    'P': [18],
    'K': [37],
    'temperature': [22.91],
    'humidity': [85.40],
    'ph': [7.13]
})

# Use the trained models to make predictions
rf_prediction = rf_classifier.predict(testing_data)
rnn_prediction = np.argmax(model.predict(np.expand_dims(testing_data, axis=2)), axis=1)

# Combine the predictions using a simple averaging method
combined_prediction = (rf_prediction + rnn_prediction) / 2

# Convert the combined prediction to integer indices
combined_prediction_indices = combined_prediction.astype(int)

# Map the combined prediction indices to the corresponding crop names using the label encoder
predicted_crop_name = label_encoder.inverse_transform(combined_prediction_indices)

# Output the predicted crop name
print("Predicted Crop Name:", predicted_crop_name)