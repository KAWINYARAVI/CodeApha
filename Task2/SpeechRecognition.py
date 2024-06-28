import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Assuming you have a dataset with pre-extracted MFCC features and labels
# Replace with actual dataset loading and preprocessing steps

# Dummy example: Generating random data
# Replace with actual dataset loading and preprocessing steps
def load_data():
    # Dummy example: Generating random data
    X = np.random.randn(100, 20, 44)  # 100 samples of MFCCs (20 frames, 44 MFCC coefficients)
    y = np.random.randint(0, 3, 100)  # Emotion labels (0, 1, 2)
    return X, y

# Load data
X, y = load_data()

# Split data into train, validation, test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Reshape for CNN input (add channel dimension for grayscale images)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Assuming 3 emotion classes
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Predictions
predictions = model.predict(X_test)

# Example of predicting emotions for new data (dummy example)
# Replace with actual prediction logic for your application
new_data = np.random.randn(1, 20, 44)  # Example of new MFCC data
new_data = new_data[..., np.newaxis]
predicted_class = np.argmax(model.predict(new_data), axis=-1)
print(f'Predicted emotion class: {predicted_class}')
