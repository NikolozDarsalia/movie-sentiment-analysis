# 🎬 Movie Review Sentiment Analysis: Traditional & Pre-Transformer NLP

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%7C%20Word2Vec%20%7C%20LSTM-brightgreen)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20%26%20Visualization-orange)

> **🎓 Academic Project:** This repository was developed as an assignment for the **Advanced Natural Language Processing** course. The core objective of this project is to explore, implement, and evaluate classic, "old-fashioned" (pre-transformer) NLP solutions for sentiment analysis, tracking the historical progression of the field from raw linguistics to sequence modeling.

A comprehensive Natural Language Processing (NLP) pipeline designed to classify the sentiment of movie reviews as **Positive** or **Negative**. 

## 🚀 Key Features & Progression

### 1. Rule-Based Heuristics Engine
Before applying machine learning, a purely linguistic rule-based classifier was built to establish a baseline:
* **Hard & Soft Rules:** Exact pattern matching for strong polarity vs. vote-counting for general sentiment.
* **Negation Scope Window:** A custom algorithm that tracks negation markers (`not`, `never`, `n't`) to flip the polarity of subsequent sentiment words within a specified token window.
* **Concessive Adjustments:** Logic to detect contrastive conjunctions (e.g., *"but"*, *"however"*) to appropriately penalize or boost sentiment based on the sentence's final clause.

### 2. Traditional Machine Learning (Robust Baselines)
To bridge the gap between brittle rule-based systems and complex neural networks, traditional, highly interpretable Machine Learning approaches were implemented:
* **TF-IDF with Logistic Regression:** A sparse-vector approach that evaluates the frequency and uniqueness of words across the corpus. This serves as a highly robust, mathematically interpretable baseline that easily identifies the most heavily weighted sentiment tokens.
* **Word2Vec with Logistic Regression:** A transition into dense representations. By averaging pre-trained Word2Vec embeddings for each review, the model captures semantic meaning and context rather than just word frequencies, passed through a Logistic Regression classifier for stable predictions.

### 3. Deep Learning with PyTorch
A custom recurrent neural network architecture tailored for sequence classification:
* **Bidirectional LSTM:** Processes reviews both forwards and backwards to capture maximum context.
* **Global Max Pooling:** Extracts the strongest sentiment signals from anywhere in the review, completely bypassing the "post-padding amnesia" problem common in standard RNNs.
* **Label Smoothing (SmoothBCE):** Custom Binary Cross-Entropy loss implementation to prevent the model from becoming overconfident and memorizing training noise.
* **Regularization:** Heavy implementation of Dropout (embedding and dense layers), and Weight Decay (L2) to ensure robust generalization.

### 4. Task-Specific Domain Adaptation (Fine-Tuning)
Instead of relying solely on generic pre-trained Word2Vec embeddings, the top 5,000 most frequent words in the vocabulary were **unfrozen** during Phase 2 of training. 
Using a custom gradient hook, the LSTM surgically fine-tuned these specific embeddings. This allowed generic adjectives to shift in the latent space, building a movie-specific "sentiment dictionary" without destroying the foundational grammar of the embeddings.

## 📊 Visualizing Semantic Shift

To prove the effectiveness of the fine-tuning phase, the project includes a **PCA visualization** tracking the geometric movement of word vectors.

As shown in our analysis, the binary cross-entropy loss successfully forced a **Sentiment Polarization**. Words that were previously entangled in the generic Word2Vec space were actively pulled apart into distinct positive and negative clusters.

## 📈 Results (LSTM Model)

* **Phase 1 (Frozen Embeddings):** ~78.4% Test Accuracy
* **Phase 2 (Fine-Tuned Top-K Embeddings):** ~81.3% Test Accuracy

The +3% performance bump validates the domain adaptation step, proving that allowing the model to separate antonyms in the embedding space yields superior classification boundaries.

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/NikolozDarsalia/movie-sentiment-analysis.git](https://github.com/NikolozDarsalia/movie-sentiment-analysis.git)
   cd movie-sentiment-analysis
   ```
2. **Install dependencies:**
   ```
   uv sync
   ```
3. **Run the Notebook:**
   Open the Jupyter Notebook (```sentiment_analysis.ipynb```) to explore the EDA, rule-based approach, TF-IDF/Word2Vec baselines, model training loop, and the PCA visualizations.


## 🧠 Technologies Used
- PyTorch (Model Architecture, Custom Loss, Gradient Hooks)
- Scikit-learn (TF-IDF, Logistic Regression, PCA Dimensionality Reduction)
- Regex / Base Python (Linguistic Rules, Negation Handling)
- Matplotlib / Seaborn (Data Visualization)
