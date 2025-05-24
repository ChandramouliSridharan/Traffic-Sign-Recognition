import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    print(f"Image Dimension: {images.shape}")
    print ("---------------------------------------------------------------")

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(images,labels, test_size=TEST_SIZE)

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model_history = model.fit(x_train, y_train, epochs=EPOCHS,validation_data=(x_test, y_test))

    # Print training and validation accuracy for the final epoch
    final_train_accuracy = model_history.history['accuracy'][-1]
    print(f"Final Training Accuracy: {final_train_accuracy:.4f}")

    # Evaluate neural network performance
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot the train and test accuracy across each Epochs.
    plot_training_history(model_history, EPOCHS)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []
    
    print("Loading Dataset...")
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        for filename in os.listdir(category_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.ppm')):
                continue  

            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue  
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(category)

    return np.array(images), np.array(labels)

def get_model():
    model = tf.keras.Sequential()

    # First Conv2D layer with Batch Normalization and MaxPooling
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Second Conv2D layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output for the Dense layers
    model.add(tf.keras.layers.Flatten())

    # Fully connected Dense layers with Dropout
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))

    # Output layer with NUM_CATEGORIES units and softmax activation
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])
    return model

def plot_training_history(history, epochs):
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs_range, history.history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
