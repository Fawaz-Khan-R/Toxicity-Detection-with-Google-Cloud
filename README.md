# Toxicity Detection and Mitigation Model

## Overview
This repository contains the code and resources for a toxicity detection model developed during the Google Cloud Skills Boost course on Responsible AI for Developers. The project applies machine learning techniques to identify harmful language in text, emphasizing ethical AI practices. The model uses a simple Keras sequential architecture with initial embedding and dense layers, outputting toxicity predictions. Additionally, it employs the MinDiff method to mitigate bias in the model's predictions.

## Model Evaluation Metrics

| Metric                | Value                      |
|-----------------------|----------------------------|
| Validation Accuracy    | 0.9216                     |
| Loss                   | 0.2316                     |
| Nonsensitive FPR (Threshold = 0.3) | 0.06023           |
| Sensitive FPR (Threshold = 0.3)    | 0.14778           |

## Responsible AI
Responsible AI refers to the development and deployment of artificial intelligence systems that are ethical, fair, and accountable. It encompasses principles that guide the design of AI technologies to ensure they do not perpetuate harm or discrimination. Key aspects include:

- **Fairness**: Ensuring that AI systems treat all individuals equitably, without discrimination based on race, gender, or other sensitive characteristics.
- **Transparency**: Providing clear explanations of how AI systems make decisions, allowing users to understand and trust their outputs.
- **Accountability**: Establishing mechanisms to hold developers and organizations responsible for the outcomes of their AI systems.

### Fairness and Bias
Bias in AI can arise from various sources, including biased training data, algorithmic design choices, and human influence during model development. In the context of toxicity detection, it is crucial to address bias to avoid unfair treatment of certain groups or individuals. For example, if a model is trained on data that disproportionately represents one demographic, it may perform poorly for others.

#### Impact of Bias in Toxicity Detection
- **Discriminatory Outcomes**: A biased model may incorrectly label non-toxic language as toxic for specific groups, leading to negative consequences such as censorship or exclusion.
- **Reinforcement of Stereotypes**: If the training data reflects societal biases, the model may perpetuate harmful stereotypes by misclassifying language associated with certain demographics.

To address these challenges, this project implements the MinDiff technique within TensorFlow to help mitigate bias in toxicity predictions.

## Libraries Used
This project utilizes several libraries for data processing, modeling, and evaluation:
```python
import numpy as np
import seaborn as sns
import tensorflow as tf
import copy
import tensorflow_hub as hub
import tensorflow_model_remediation.min_diff as md
from tensorflow_model_remediation.tools.tutorials_utils import (
    min_diff_keras_utils,
)
```
## Model Architecture
The model is built using Keras and consists of:
- **Embedding Layer**: Transforms input text into dense vectors.
- **Dense Layers**: Processes the embedded text data to produce toxicity predictions.
- **Optimizer**: The Adam optimizer is used for training due to its efficiency in handling sparse gradients and its adaptive learning rate capabilities.

### MinDiff Method
The MinDiff method is employed to reduce bias in model predictions by adjusting the training process. It aims to ensure that the model performs equitably across different sensitive groups by minimizing discrepancies in prediction outcomes.

## Jupyter Notebook Introduction
In this notebook, we’ll train a text classifier to identify written content that could be considered toxic or harmful and apply MinDiff to remediate some fairness concerns. In our workflow, we will:
1. Train and evaluate our baseline model’s performance on text containing references to sensitive groups.
2. Improve performance on any underperforming groups by training with MinDiff.
3. Evaluate the new model’s performance on our chosen metric.

*The purpose of this notebook is to demonstrate the usage of the MinDiff technique with a minimal workflow. It does not lay out a comprehensive approach to fairness in machine learning; our evaluation focuses only on one sensitive category and a single metric. We also do not address potential shortcomings in the dataset nor tune our configurations in this notebook.*

## Evaluation Metrics
To assess the performance of our toxicity detection model, we calculate the False Positive Rate (FPR) for both sensitive and non-sensitive groups based on a threshold of 0.3:

- **Nonsensitive FPR**: 0.06870
- **Sensitive FPR**: 0.06957

These results indicate that both sensitive and nonsensitive groups have relatively similar false positive rates, which is crucial for ensuring fairness in toxicity detection.

## Future Improvements
To enhance this project further, the goals to be implemented are:
- Implement hyperparameter tuning techniques such as grid search or random search for optimizing model performance.
- Introduce regularization techniques like dropout layers to prevent overfitting.
- Explore advanced architectures like LSTM or transformers for improved handling of sequential text data.
- Consider ensemble methods by combining predictions from multiple models for better accuracy.

## ⚠️Important Warning

**Please note:** If you run the `min_diff_keras.ipynb` file, you must do so in a Google authorized Vertex AI Workbench environment. Running this notebook outside of Vertex AI Workbench may result in dependency errors due to the proprietary nature of the `tensorflow_model_remediation.min_diff` and `min_diff_keras_utils` modules, which are specifically designed for use within Google Cloud's infrastructure.

To access and run the notebook, please implement it in Google Cloud's Vertex AI Workbench.
