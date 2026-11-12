# Emotion Classification with Attention Mechanisms

A deep learning project for multi-class emotion classification across **6 core emotions** (Joy, Sadness, Anger, Fear, Surprise, Disgust) using recurrent sequence models with Bahdanau attention mechanisms, compared against a fine-tuned Hugging Face Transformer (DistilBERT).

---

## 📌 Project Overview

Understanding emotional nuance in text is critical for customer sentiment analysis, mental health monitoring, and conversational AI. This project addresses multi-class emotion detection by:
1. **Mapping Fine-Grained Emotions to Core Ekman Classes:** Converting GoEmotions (28 raw classes) into Paul Ekman's 6 basic emotions.
2. **Transfer Learning via Word Embeddings:** Utilizing pre-trained **GloVe 100d** vectors to initialize sequence model embedding representations.
3. **Comparing Recurrent Architectures:** Implementing and benchmarking standard **LSTM**, **GRU**, and **BiLSTM + Attention**.
4. **Attention Mechanism & Interpretability:** Formulating an additive Bahdanau self-attention mechanism over BiLSTM hidden states that outputs token-level attention weights for model explainability.
5. **Transformer Fine-Tuning:** Leveraging Hugging Face's `Trainer` API to fine-tune `distilbert-base-uncased`.
6. **Interactive Web Deployment:** Providing a interactive **Streamlit** application featuring real-time emotion prediction, class confidence distributions, and interactive attention heatmap visualizers.

---

## 🏗️ Model Architectures

### 1. LSTM & GRU Baseline Classifiers
- **Embedding Layer:** Initialized with pre-trained GloVe 100d vectors.
- **Recurrent Layer:** 2-layer LSTM or GRU with hidden dimension of 128.
- **Classification Head:** Dropout (0.3) -> Linear (128 -> 64) -> ReLU -> Dropout -> Linear (64 -> 6).

### 2. BiLSTM + Bahdanau Attention
- **Bidirectional Recurrence:** 2-layer BiLSTM producing forward and backward hidden representations $H \in \mathbb{R}^{T \times 2D}$.
- **Additive Attention Scoring:**
  $$u_t = \tanh(W_a h_t + b_a)$$
  $$\alpha_t = \frac{\exp(v_a^\top u_t)}{\sum_{k=1}^T \exp(v_a^\top u_k)}$$
- **Dynamic Context Vector:**
  $$c = \sum_{t=1}^T \alpha_t h_t \in \mathbb{R}^{2D}$$
- **Padding Masking:** Zero tokens (`<pad>`) are masked out with $-\infty$ energy scores so they receive zero attention weight.
- **Output:** Returns both class logits and per-token attention distribution $\alpha$ for interpretability heatmaps.

### 3. Fine-Tuned DistilBERT (Transformer)
- Pre-trained `distilbert-base-uncased` with 6-class classification head.
- Optimized with AdamW, linear learning rate warmup, and macro F1 checkpointing via Hugging Face `Trainer`.

---

## 🎯 Emotion Mapping (GoEmotions to Ekman 6)

| Core Emotion | Mapped GoEmotions Raw Classes |
| :--- | :--- |
| **Joy** | `admiration`, `amusement`, `approval`, `excitement`, `gratitude`, `joy`, `love`, `optimism`, `pride`, `relief` |
| **Sadness** | `disappointment`, `embarrassment`, `grief`, `remorse`, `sadness` |
| **Anger** | `anger`, `annoyance`, `disapproval` |
| **Fear** | `fear`, `nervousness` |
| **Surprise** | `confusion`, `curiosity`, `realization`, `surprise` |
| **Disgust** | `disgust` |

---

## 📁 Repository Structure

```
├── app.py                                  # Interactive Streamlit Web Application
├── train_all_and_compare.py                # End-to-end training, evaluation & benchmark script
├── emotion_classification_with_attention.ipynb # Interactive Jupyter Notebook
├── requirements.txt                        # Python dependencies
├── src/
│   ├── dataset.py                          # Data loading, Ekman mapping, cleaning, GloVe loader
│   ├── attention.py                        # Bahdanau attention mechanism module
│   ├── models.py                           # LSTM, GRU, BiLSTM+Attention PyTorch models
│   ├── train.py                            # PyTorch training loop & early checkpointing
│   ├── train_transformer.py                # Hugging Face Trainer DistilBERT fine-tuning
│   ├── evaluate.py                         # Accuracy, Macro F1, Confusion Matrix generator
│   └── visualize.py                        # Attention heatmap & comparison plotting
├── saved_models/                           # Checkpoints (.pt, vocab.json, distilbert)
├── results/                                # Confusion matrices, heatmaps & comparison CSV
├── data/                                   # Cleaned train/val/test CSV splits
├── Simplified/                             # GoEmotions Parquet source files
└── raw/                                    # Raw GoEmotions parquet
```

---

## 🚀 Getting Started

### 1. Installation
Clone or navigate to the directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train All Models and Generate Benchmarks
Run the complete training pipeline:
```bash
python train_all_and_compare.py
```
This script will:
- Process the GoEmotions dataset into 6 Ekman classes.
- Build vocabulary and load pre-trained GloVe 100d vectors.
- Train and evaluate LSTM, GRU, and BiLSTM + Attention.
- Fine-tune DistilBERT using Hugging Face Trainer.
- Save confusion matrix plots and `results/model_comparison.csv`.
- Generate attention heatmaps on representative samples.

### 3. Launch Interactive Streamlit App
Start the web application:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🔍 Attention Explainability

The BiLSTM + Attention model provides direct insight into why a specific emotion was predicted. For example:
- Input: *"I was completely stunned and never expected that plot twist!"*
  - Top weighted tokens: **"stunned"** (62%), **"expected"** (19%) $\rightarrow$ **Surprise**
- Input: *"The smell of that rotten food was repulsive and sickening."*
  - Top weighted tokens: **"sickening"** (48%), **"repulsive"** (31%), **"rotten"** (15%) $\rightarrow$ **Disgust**

---

## 📊 Evaluation Metrics

Models are evaluated across:
- **Accuracy**
- **Macro F1** (unweighted mean across all 6 classes, ensuring minority classes like Fear/Disgust are prioritized)
- **Weighted F1**
- **Per-Class Precision, Recall, and F1-Score**
- **Confusion Matrix Heatmaps**

