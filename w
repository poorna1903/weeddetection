from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

weed_dir = '/content/drive/My Drive/Colab Notebooks/weed (1)/'
nonweed_dir='/content/drive/My Drive/Colab Notebooks/crop/'

# Function to load images from folder and assign labels
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to 128x128
            images.append(img)
            labels.append(label)
    return images, labels

# Load weed and non-weed images
weed_images, weed_labels = load_images_from_folder(weed_dir, 1)
non_weed_images, non_weed_labels = load_images_from_folder(nonweed_dir, 0)

# Combine datasets
images = np.array(weed_images + non_weed_images)
labels = np.array(weed_labels + non_weed_labels)

# Normalize images
images = images / 255.0

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=12, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    if prediction[0] >0.5:
        return "Weed"
    else:
        return "Non-Weed"

model.summary() #This might be a extra procedure which will just explain the program not manipulate in the accuracy


#Loading and depecting the model(testing the model after it is completely been trained)
import os
from PIL import Image

new_image_path = 'C3TM.jpg'
label = predict_image(new_image_path)
print(f"The image is labeled as: {label}")

# Function to predict and label image blocks
def block_predict(image, block_size=(128, 128)):
    height, width, _ = image.shape
    labeled_image = image.copy()
    for y in range(0, height, block_size[1]):
        for x in range(0, width, block_size[0]):
            block = image[y:y+block_size[1], x:x+block_size[0]]
            if block.shape[0] != block_size[1] or block.shape[1] != block_size[0]:
                continue  # Skip blocks that are not the correct size
            block_normalized = block / 255.0
            block_normalized = np.expand_dims(block_normalized, axis=0)
            prediction = model.predict(block_normalized)
            label = "non-Weed" if prediction[0] > 0.5 else "Weed"
            color = (0, 0, 255) if label == "Weed" else (0, 255, 0)
            cv2.putText(labeled_image, label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(labeled_image, (x, y), (x + block_size[0], y + block_size[1]), color, 2)
    return labeled_image

# Test the model on a new image
new_image_path = 'C3TM.jpg'
image = cv2.imread(new_image_path)
labeled_image = block_predict(image)


# Save and display the labeled image
cv2.imwrite('labeled_image.jpg', labeled_image)
plt.imshow(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axis
plt.show()

new_image_path = 'IMG_20240714_111435.jpg'
image = cv2.imread(new_image_path)
labeled_image = block_predict(image)

# Save and display the labeled image
cv2.imwrite('labeled_image.jpg', labeled_image)
plt.imshow(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axis
plt.show()

#Loading the model after the model is trained and the desired test result is obtained
model.save('weed.h5')
