import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from sklearn.model_selection import train_test_split

# Load  data 
data = pd.read_csv("")
x = data.iloc[:, :-1] 
y = data.iloc[:, -1]

# split data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalize or scale the input data if necessary
scalar = StandardScaler()
x_train_scaled = scalar.fit_transform(x_train)
x_val_scaled = scalar.fit_transform(x_val)

# Constants and hyperparameters
sequence_length = x_train.shape[1]
num_features = x_train.shape[2]
vocabulary_size = len(np.unique(y_train))

# Define the CNN model
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, num_features)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=256, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten()
])

# Define the RNN model with LSTM layers
rnn_model = Sequential([
    LSTM(64, input_shape=(sequence_length, num_features), return_sequences=True),
    LSTM(64)
])

# Combine the CNN and RNN models into a hybrid model
combined_model = Sequential([
    concatenate([cnn_model.output, rnn_model.output]),  # Concatenate CNN and RNN outputs
    Dense(256, activation='relu'),
    Dense(vocabulary_size, activation='softmax')
])

# Compile the hybrid model
combined_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display model summary
combined_model.summary()

# Train the hybrid model
combined_model.fit([x_train_scaled, x_train_scaled], y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = combined_model.evaluate([x_val_scaled, x_val_scaled], y_val)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Perform inference with the trained model on new sensor data
new_x = np.load('new_sensor_data.npy')  # Load your new sensor data   ######## will be input ###########
new_x_train_scaled = scalar.fit_transform(x_train)

# Make predictions using the hybrid model
cnn_features = cnn_model.predict(new_x_train_scaled)
rnn_features = rnn_model.predict(new_x_train_scaled)

# Concatenate CNN and RNN features for prediction
combined_features = np.concatenate([cnn_features, rnn_features], axis=-1)
predicted_probabilities = combined_model.predict(combined_features)
predicted_word_index = np.argmax(predicted_probabilities, axis=1)[0]

print(f"Predicted Word Index: {predicted_word_index}")