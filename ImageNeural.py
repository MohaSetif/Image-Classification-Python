import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

# Define the new resolution for the images (e.g., 128x128 pixels)
# new_resolution = (64, 64)

# # Resize the images to the new resolution
# training_images_resized = np.array([cv2.resize(img, new_resolution) for img in training_images])
# testing_images_resized = np.array([cv2.resize(img, new_resolution) for img in testing_images])

# # Normalize pixel values
# training_images_resized, testing_images_resized = training_images_resized / 255, testing_images_resized / 255

# # Create the model
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(new_resolution[0], new_resolution[1], 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(training_images_resized, training_labels, epochs=10, validation_data=(testing_images_resized, testing_labels))

# # Evaluate the model
# loss, accuracy = model.evaluate(testing_images_resized, testing_labels)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")

# # Save the model
# model.save("image_classifier.model")


model = models.load_model('image_classifier.model')

img = cv2.imread('Dog 64x64.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)

print(f"Prediction: {class_names[index]}")

plt.show()