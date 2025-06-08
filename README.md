# Human Activity Recognition with PCA and MLP

This project focuses on classifying human physical activities using sensor data by applying **dimensionality reduction with PCA** and building a **Multilayer Perceptron (MLP)** for accurate classification.

## 1. Data Preprocessing

The dataset was loaded and explored with descriptive statistics and visualizations. The target variable, `"Activity"`, represents various human activities (e.g., walking, sitting, etc.).

- Checked for null values (none found)
- Explored data distributions using seaborn/matplotlib
- Plotted acceleration averages grouped by activity

---

## 2. Unsupervised Analysis: PCA

To reduce the high dimensionality of the dataset:

- **Normalization** was applied to numeric features.
- **One-hot encoding** was applied to the categorical target.
- **Principal Component Analysis (PCA)** reduced the feature space to **64 components**, explaining **90%** of the variance.

---

## 3. Modeling: Multilayer Perceptron (MLP)

A deep learning classification model using TensorFlow/Keras:

- **Input Layer:** 64 features
- **Hidden Layers:** 2 fully connected layers with ReLU activation
- **Regularization:** L2 and **Dropout** added to prevent overfitting
- **Output Layer:** Softmax activation for multiclass classification
- **Optimizer:** Adam with learning rate = 0.001
- **Epochs:** 50
- **Batch size:** 32

---

## 4. Evaluation

- **Accuracy:** 96%
- **Classification report:** High precision, recall, and F1-score across all classes
- Dropout helped stabilize class-wise predictions

---

## 5. Visualizations

- **Learning Curves**: Accuracy and loss plotted for both training and validation
- **PCA Scatter Plot**: Activity clusters in reduced dimensional space

