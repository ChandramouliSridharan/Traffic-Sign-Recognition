# Traffic Sign Classification Project

This project involves the classification of traffic signs using a Convolutional Neural Network (CNN) trained on the German Traffic Sign Recognition Benchmark (GTSRB)
dataset. The model is designed to accurately identify 43 different categories of traffic signs based on image inputs, providing a robust solution for 
real-world applications like autonomous driving and driver assistance systems.

## Project Overview

The implementation uses TensorFlow and Keras to build and train a CNN. The dataset contains images in `.ppm` format, which are resized to 30x30 pixels for uniform input dimensions. 
The neural network consists of convolutional layers for feature extraction, batch normalization for stable training, dropout layers to reduce overfitting, and dense layers to classify the traffic signs.

## Experimentation Process

### Initial Attempts

The experimentation began with a baseline CNN architecture consisting of two convolutional layers, each followed by max-pooling layers, and a dense layer of 128 neurons. While this model achieved a training accuracy of 55% and a test accuracy of 68%, there was noticeable overfitting, test performance suggested room for improvement, potential underutilization of available features due to fewer neurons in the dense layer.

### Optimizations

To address these issues, the following changes were made:

1. **Batch Normalization:** Added after each convolutional layer to normalize activations, stabilize training, and improve convergence speed. 
   
2. **Revised Dense Layers:** Increased the dense layer size to 256 neurons to better capture complex patterns in the data. Additionally, a previously used dense layer of 126 neurons was removed to streamline the architecture.

3. **Reduced Learning Rate:** Lowered the learning rate of the Adam optimizer to 0.0001, allowing more controlled weight updates and leading to a smoother training process.

These optimizations resulted in a significant improvement, with the final model achieving **97.65% training accuracy** and **98.70% test accuracy**, demonstrating robust generalization.

### Observations

- **What Worked Well:** 
   - Batch normalization played a crucial role in improving both training accuracy and test accuracy.
   - Reducing the learning rate improved test accuracy and prevented overfitting.
   - The streamlined architecture reduced complexity and improved generalization without loss of accuracy.

- **What Didn’t Work Well:**
   - The initial model suffered from overfitting due to insufficient regularization and lack of normalization.
   - Adding more dense layers in early experiments increased model complexity but did not enhance test accuracy.
     
## How to Run the Project

### Command to Execute the Script:

To train the model and save it, run the following command in the terminal:

***python .\traffic.py <path_to_the_dataset_directory> model.h5***


### Notes:
- Ensure all required Python packages are installed, including TensorFlow, NumPy, OpenCV, and scikit-learn.
- The dataset directory should contain subfolders named `0` to `42`, each holding the respective traffic sign images in `.ppm` format.

---

## Conclusion

This project demonstrates the importance of iterative optimization in deep learning. Starting with a basic model, careful tuning of architecture, learning rates, and regularization strategies led to a highly accurate and generalizable traffic sign classifier.