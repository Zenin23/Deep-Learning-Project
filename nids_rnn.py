# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
attack_data = pd.read_csv('CTU13_Attack_Traffic.csv')
normal_data = pd.read_csv('CTU13_Normal_Traffic.csv')

# Add labels: 1 for attack, 0 for normal
attack_data['Label'] = 1
normal_data['Label'] = 0

# Concatenate both datasets
data = pd.concat([attack_data, normal_data], ignore_index=True)

# Drop missing values
data.dropna(inplace=True)

# Assume 'Timestamp' or similar non-numeric columns are dropped
# Drop columns not useful for modeling (adjust as per dataset)
data = data.select_dtypes(include=[np.number])

# Separate features and target label
X = data.drop('Label', axis=1)
y = data['Label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input to 3D for LSTM: [samples, time steps, features]
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.3, random_state=42)


# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))


# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# Marking predictions
results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred.flatten()})
results["Notation"] = results["Predicted"].apply(lambda x: "Attack" if x == 1 else "Normal")
print(results.head(10))


# Calculate the correlation matrix
correlation_matrix = data.corr()

# Display the correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap for CTU-13 Dataset')
plt.show()


# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Initialize the RNN model
model = Sequential()

# Input Layer (LSTM layer)
model.add(LSTM(units=64, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Dropout to prevent overfitting

# Adding more LSTM layers for better representation
model.add(LSTM(units=128, activation='tanh', return_sequences=False))
model.add(Dropout(0.2))

# Batch normalization (optional, helps stabilize training)
model.add(BatchNormalization())

# Fully connected (Dense) layer
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.3))

# Output Layer (binary classification)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()