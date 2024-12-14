# Bird Classification

This project focuses on classifying various bird species using deep learning techniques. By leveraging convolutional neural networks (CNNs), the model aims to accurately identify bird species from images, contributing to research in ornithology and aiding bird enthusiasts.The dataset contains 70626 images belonging to 450 classes. InceptionV3 architecture is used to implement this deep learning model. The accuracy of this model is 91%.

## Dataset

The dataset comprises images of different bird species, each labeled accordingly. Key characteristics include:

- **Number of Classes:** 450 bird species.
- **Total Images:** 70,626 images.
- **Image Resolution:** 224x224 pixels.

The dataset is sourced from a collection designed for fine-grained image classification tasks, offering extensive diversity for robust model training.

## Model Architecture

The classification model is built using the InceptionV3 architecture, which is known for its efficiency and accuracy in image classification tasks. Key features of the model include:

- **Convolutional Layers:** Extract spatial features from input images.
- **Pooling Layers:** Reduce dimensionality and retain essential features.
- **Fully Connected Layers:** Perform classification based on extracted features.
- **Activation Functions:** Utilize ReLU for non-linearity and Softmax for output probabilities.

Transfer learning is employed by fine-tuning the pre-trained InceptionV3 model for improved performance.

## Training

- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Batch Size:** 32
- **Number of Epochs:** 50

Data augmentation techniques such as random cropping, flipping, and rotation are applied to improve model generalization and robustness.

## Evaluation

The model's performance is evaluated using:

- **Accuracy:** Achieved a test accuracy of 91%.
- **Confusion Matrix:** Illustrates the model's classification performance across different classes.
- **Precision, Recall, F1-Score:** Detailed metrics for each class to assess overall effectiveness.

Visualization of accuracy and loss curves is provided to analyze model performance over training epochs.

## Prerequisites

- **Programming Language:** Python 3.x
- **Libraries:** TensorFlow/Keras, NumPy, Pandas, Matplotlib, scikit-learn

Ensure all dependencies are installed using the `requirements.txt` file provided.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shriya-1603/bird_classification.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd bird_classification
   ```
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the dataset:**
   - Ensure the dataset is properly structured and preprocessed for training.

2. **Train the model:**
   ```bash
   python train.py
   ```
   Adjust hyperparameters in the `train.py` script as needed.

3. **Evaluate the model:**
   ```bash
   python evaluate.py
   ```
   This will generate evaluation metrics and visualizations.

4. **Make predictions:**
   ```bash
   python predict.py --image_path path_to_image
   ```
   Replace `path_to_image` with the actual path to the image you want to classify.

The confusion matrix and classification reports are included for a detailed performance analysis. The model shows strong performance in distinguishing between visually similar bird species.


## Future Work

- **Model Improvement:** Explore more advanced architectures or ensemble methods to enhance accuracy.
- **Dataset Expansion:** Incorporate additional bird species and images to improve generalization.
- **Deployment:** Develop a user-friendly application or API for real-time bird species identification.

## Acknowledgements

- [Caltech-UCSD Birds-200-2011 Dataset](http://www.vision.caltech.edu/datasets/cub_200_2011/)
- Open-source libraries and pre-trained models used for transfer learning.
- Contributors and collaborators who supported the project.
