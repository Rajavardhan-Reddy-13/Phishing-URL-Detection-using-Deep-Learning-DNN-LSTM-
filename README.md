# Phishing-URL-Detection-using-Deep-Learning-DNN-LSTM-
This project aims to detect phishing URLs using two different implementation tasks. (1. Feature Engineering + DNN) and (2. LSTM)

1. **Task 1** – A feature-engineered **Deep Neural Network (DNN)** model
2. **Task 2** – A sequence-based **LSTM (Long Short-Term Memory)** model

The goal is to identify whether a given URL is **legitimate or phishing** based on patterns either extracted manually (Task 1) or learned automatically (Task 2).

---

## 📂 Dataset Information

The dataset used is a modified version of the one from the following Kaggle project:

> 🔗 [Phishing Detection using ML Techniques – by Mohamed Galal](https://www.kaggle.com/code/mohamedgalal1/phishing-detection-using-ml-techniques)

### Dataset Structure:
- Contains **507,195 URLs** labeled as:
  - `good` (legitimate)
  - `bad` (phishing)
- Each row includes:
  - A **URL string**
  - A **label** indicating if it is phishing (`bad`) or not (`good`)

---

## 🧠 Project Overview

### ✅ Task 1: Feature-Based Deep Neural Network (DNN)

In this approach, we manually extracted features from URLs to train a traditional feed-forward neural network.

#### Features Extracted:
1. Whether the URL contains an **IP address**
2. **Number of dots** in the URL (more dots can indicate subdomain abuse)
3. **Total length** of the URL (longer URLs may try to hide intentions)
4. **Domain age** (calculated via WHOIS lookup)
5. Whether the URL contains a **redirection (//)**
6. Presence of **JavaScript** code in the URL
7. Count of **special characters** (e.g., `@`, `%`, `$`) in the URL

#### DNN Architecture:
- 2 Hidden Layers using LeakyReLU + BatchNorm
- Output Layer using Sigmoid activation
- Trained with Binary Cross-Entropy Loss and Adam Optimizer

---

### ✅ Task 2: Sequence-Based LSTM Model

This model processes raw URLs as sequences of characters and lets the LSTM learn patterns without manual feature extraction.

#### Preprocessing Steps:
- **Character-level tokenization** using Keras `Tokenizer`
- **Padding** each sequence to a fixed length (200 characters)
- Convert to PyTorch tensors and build a custom Dataset + DataLoader

#### LSTM Architecture:
- `Embedding` Layer: Converts each character index into a dense vector
- `LSTM` Layer: Captures sequential patterns in character order
- `Linear + Sigmoid` Output: Outputs phishing probability

---

#### References:
- https://www.kaggle.com/code/habibmrad1983/habib-mrad-detection-malicious-url-using-ml-models
- https://www.kaggle.com/code/mohamedgalal1/phishing-detection-using-ml-techniques/data
- Title: Classifying Phishing URLs Using Recurrent Neural Networks. Link for this paper: 
  https://albahnsen.github.io/files/Classifying%20Phishing%20URLs%20Using%20Recurrent%20Neural%20Networks_cameraready.pdf 
- Word embedding: https://www.tensorflow.org/text/guide/word_embeddings 
- PyTorch LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html 
- LSTM video: https://wandb.ai/sauravmaheshkar/LSTM-PyTorch/reports/Using-LSTM-in-PyTorch-A-Tutorial-With-Examples--VmlldzoxMDA2NTA5 
- A research paper that uses LSTM to detect phishing websites: https://onlinelibrary.wiley.com/doi/full/10.1002/spy2.256

---
