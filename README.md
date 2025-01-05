# Covid-19-CNN: Classifying COVID-19 Patients Using CNN from Lung X-rays

This project demonstrates the development of a Convolutional Neural Network (CNN) for classifying COVID-19 patients based on lung X-ray images. The workflow involves preparing the dataset, building the model, training it, and evaluating its performance. This project was implemented using **Google Colab** for efficient and accessible computation.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Workflow](#model-workflow)
- [Results](#results)
- [Contributing](#contributing)

---

## Project Overview

This project applies deep learning techniques to classify lung X-rays as COVID-19 positive or negative. The primary focus is on leveraging CNNs to extract meaningful patterns from X-ray images and make accurate predictions.

**Key highlights:**
- Dataset includes X-ray images of lungs, labeled as COVID-19 positive or negative.
- Implemented and executed using **Google Colab** to leverage its free GPU resources.
- Built with Python and TensorFlow/Keras for CNN model implementation.

---

## Features

- Automated dataset downloading and preparation.
- Custom CNN architecture tailored for medical image classification.
- Training with real-time accuracy and loss visualization.
- Evaluation metrics to validate model performance, such as accuracy and confusion matrix.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/CV19CNN.git
   cd CV19CNN
   ```
2. Install the required dependencies
   ```bash
   pip install -r requirements.txt
   ```
---

## Usage

1.	Open the notebook in [Google Colab](https://colab.research.google.com/drive/1j54DzfeveQmQ1C3AuzoUF4pyHhytvuGO?usp=sharing)
2.	Download the dataset:
- Use the script in the notebook to download the dataset automatically.
-	Dataset contains lung X-ray images labeled for COVID-19 detection.
3.	Run the Jupyter Notebook:
-	Execute the cells in the notebook sequentially for:
-	Dataset preparation.
-	Model training.
-	Evaluation and visualization of results.

---

## Model Workflow
**1.	Dataset Preparation**
The dataset of lung X-rays is preprocessed to ensure proper resizing and normalization for input into the CNN. Train-test splits are applied to evaluate model performance.

**2.	Model Creation**
A CNN is designed with layers including convolutional, pooling, and dense layers optimized for image classification tasks.

**3.	Model Training**
The model is trained using the Adam optimizer and categorical crossentropy loss function. Training performance is monitored through metrics such as accuracy and loss.

**4.	Model Evaluation**
Evaluate the model using metrics like accuracy, precision, recall, and a confusion matrix. Visualization of model predictions is included to assess its performance.

---

## Results

**Model's accuracy**

<img width="598" alt="image" src="https://github.com/user-attachments/assets/1a74b0a5-671d-493e-892b-c7c2aedb7e3e" />

**Confusion Matrix**

![image](https://github.com/user-attachments/assets/92e2a63d-97f7-41bd-b386-4b5215ca7292)

### Examples

![image](https://github.com/user-attachments/assets/b803377d-28ee-4bb1-a49c-27ad726e26d8)
![image](https://github.com/user-attachments/assets/84b29b60-508e-4d51-ad6e-b8e180bf255f)

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any features or improvements.


