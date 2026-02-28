import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

IMG_SIZE = 128
data = []
labels = []

# Absolute dataset path (IMPORTANT)
dataset_path = os.path.join(os.getcwd(), "dataset")

print("Looking for dataset at:", dataset_path)

if not os.path.exists(dataset_path):
    print("ERROR: dataset folder not found!")
    exit()

for label, category in enumerate(["fake", "real"]):
    path = os.path.join(dataset_path, category)

    if not os.path.exists(path):
        print(f"ERROR: {category} folder not found inside dataset!")
        exit()

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(label)

if len(data) == 0:
    print("ERROR: No images found in dataset folders!")
    exit()

data = np.array(data) / 255.0
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Starting training...")
model.fit(X_train, y_train, epochs=10)

# --- NEW: Check how well the model actually performs ---
print("\nEvaluating model on test data...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%\n")

# --- THE FIX: Save as .h5 format ---
model.save("model.h5")
print("Model saved successfully as model.h5!")