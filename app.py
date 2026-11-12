import os
import json
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import EMOTIONS, ID_TO_EMOTION, EMOTION_TO_ID, clean_text, Vocabulary
from src.models import LSTMClassifier, GRUClassifier, BiLSTMAttentionClassifier
from src.visualize import get_highlighted_html, plot_attention_heatmap

# Page configuration
st.set_page_config(
    page_title="Emotion Classification with Attention",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .dominant-card {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .dominant-title {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .dominant-emotion {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .dominant-confidence {
        font-size: 1.1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

EMOTION_META = {
    'joy': {'emoji': '😄', 'color': '#10B981', 'bg': '#ECFDF5', 'text': '#065F46'},
    'sadness': {'emoji': '😢', 'color': '#3B82F6', 'bg': '#EFF6FF', 'text': '#1E40AF'},
    'anger': {'emoji': '😡', 'color': '#EF4444', 'bg': '#FEF2F2', 'text': '#991B1B'},
    'fear': {'emoji': '😨', 'color': '#8B5CF6', 'bg': '#F5F3FF', 'text': '#5B21B6'},
    'surprise': {'emoji': '😲', 'color': '#F59E0B', 'bg': '#FFFBEB', 'text': '#92400E'},
    'disgust': {'emoji': '🤢', 'color': '#84CC16', 'bg': '#F7FEE7', 'text': '#3F6212'}
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_vocab():
    vocab_path = "saved_models/vocab.json"
    if os.path.exists(vocab_path):
        return Vocabulary.load(vocab_path)
    return None


@st.cache_resource
def load_pytorch_model(model_type: str, _vocab):
    if _vocab is None:
        return None
    vocab_size = len(_vocab)
    embedding_dim = 100
    hidden_dim = 128
    num_classes = len(EMOTIONS)

    if model_type == "BiLSTM + Attention":
        model = BiLSTMAttentionClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_classes=num_classes,
            attention_dim=128,
            dropout=0.3
        )
        weights_file = "saved_models/bilstm_attention.pt"
    elif model_type == "LSTM":
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3
        )
        weights_file = "saved_models/lstm.pt"
    elif model_type == "GRU":
        model = GRUClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3
        )
        weights_file = "saved_models/gru.pt"
    else:
        return None

    if os.path.exists(weights_file):
        model.load_state_dict(torch.load(weights_file, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    return None


@st.cache_resource
def load_transformer_pipeline():
    model_dir = "saved_models/distilbert_emotion"
    if os.path.exists(model_dir):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model
    return None, None


# Main Layout
st.markdown("<div class='main-header'>Emotion Classification with Attention Mechanisms</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Classify text across 6 core emotions (Joy, Sadness, Anger, Fear, Surprise, Disgust) and interpret token-level attention drivers.</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose Classifier Architecture:",
    [
        "BiLSTM + Attention (with Heatmap)",
        "Fine-Tuned DistilBERT (Transformer)",
        "LSTM (GloVe)",
        "GRU (GloVe)"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset & Framework")
st.sidebar.info("""
- **Dataset:** GoEmotions mapped to 6 Ekman core classes
- **Embeddings:** GloVe 100d Vectors
- **Sequence Models:** PyTorch Bidirectional LSTM & GRU
- **Attention:** Additive (Bahdanau) Attention layer
- **Transformer:** DistilBERT Fine-Tuned via HF Trainer API
""")

if os.path.exists("results/model_comparison.csv"):
    comp_df = pd.read_csv("results/model_comparison.csv")
    st.sidebar.subheader("Model Benchmark (Test Set)")
    st.sidebar.dataframe(comp_df, hide_index=True)

# Load Vocab
vocab = load_vocab()

# Sample prompts
st.subheader("1. Enter Text or Pick a Sample")
sample_cols = st.columns(6)

sample_map = {
    "Joy": "I was just offered my dream job and I cannot stop smiling, this is the best day ever!",
    "Sadness": "I feel so lonely, empty, and broken inside since the loss.",
    "Anger": "This is completely outrageous, disrespectful, and makes my blood boil!",
    "Fear": "I heard a loud sinister sound in the dark and I am shivering in panic.",
    "Surprise": "Wait, what?! I was not expecting this plot twist at all, completely stunned!",
    "Disgust": "The sight and pungent smell of that rancid garbage was utterly sickening."
}

if 'input_text' not in st.session_state:
    st.session_state['input_text'] = sample_map["Joy"]

for idx, (emo_name, sample_val) in enumerate(sample_map.items()):
    with sample_cols[idx]:
        if st.button(f"{EMOTION_META[emo_name.lower()]['emoji']} {emo_name}", use_container_width=True):
            st.session_state['input_text'] = sample_val

user_input = st.text_area(
    "Input text for emotion analysis:",
    value=st.session_state['input_text'],
    height=110,
    placeholder="Type any sentence here..."
)

predict_btn = st.button("Predict Emotion", type="primary", use_container_width=True)

if user_input.strip():
    clean_inp = clean_text(user_input)
    
    # Inference logic
    probs = None
    attention_weights = None
    tokens = None
    pred_emotion = None
    confidence = None

    if "DistilBERT" in model_choice:
        tok, tf_model = load_transformer_pipeline()
        if tf_model is not None and tok is not None:
            inputs = tok(clean_inp, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
            with torch.no_grad():
                logits = tf_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        else:
            st.warning("DistilBERT model not found yet or still training. Falling back to BiLSTM + Attention.")
            model_choice = "BiLSTM + Attention (with Heatmap)"

    if probs is None and vocab is not None:
        if "BiLSTM" in model_choice:
            model = load_pytorch_model("BiLSTM + Attention", vocab)
            if model is not None:
                ids, length, tokens = vocab.encode(clean_inp, max_len=50)
                inp = torch.tensor([ids], dtype=torch.long).to(DEVICE)
                with torch.no_grad():
                    logits, weights = model(inp, return_attention=True)
                    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                    attention_weights = weights[0].cpu().numpy()[:length]
        elif "LSTM" in model_choice and "BiLSTM" not in model_choice:
            model = load_pytorch_model("LSTM", vocab)
            if model is not None:
                ids, length, tokens = vocab.encode(clean_inp, max_len=50)
                inp = torch.tensor([ids], dtype=torch.long).to(DEVICE)
                with torch.no_grad():
                    logits = model(inp)
                    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        elif "GRU" in model_choice:
            model = load_pytorch_model("GRU", vocab)
            if model is not None:
                ids, length, tokens = vocab.encode(clean_inp, max_len=50)
                inp = torch.tensor([ids], dtype=torch.long).to(DEVICE)
                with torch.no_grad():
                    logits = model(inp)
                    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    if probs is not None:
        pred_id = int(np.argmax(probs))
        pred_emotion = ID_TO_EMOTION[pred_id]
        confidence = float(probs[pred_id])
        meta = EMOTION_META[pred_emotion]

        st.markdown("---")
        st.subheader("2. Prediction Results")

        res_col1, res_col2 = st.columns([1, 1.6])

        with res_col1:
            st.markdown(f"""
            <div class='dominant-card' style='background-color: {meta['bg']}; border: 2px solid {meta['color']};'>
                <div class='dominant-title' style='color: {meta['text']};'>Dominant Emotion</div>
                <div class='dominant-emotion' style='color: {meta['color']};'>{meta['emoji']} {pred_emotion.capitalize()}</div>
                <div class='dominant-confidence' style='color: {meta['text']};'>Confidence: <strong>{confidence:.1%}</strong></div>
            </div>
            """, unsafe_allow_html=True)

            # Details table
            df_probs = pd.DataFrame({
                "Emotion": [e.capitalize() for e in EMOTIONS],
                "Probability": probs
            }).sort_values(by="Probability", ascending=False)
            st.dataframe(
                df_probs.style.format({"Probability": "{:.1%}"}).bar(subset=["Probability"], color=meta['color']),
                hide_index=True,
                use_container_width=True
            )

        with res_col2:
            st.markdown("##### Confidence Distribution Across 6 Core Emotions")
            fig, ax = plt.subplots(figsize=(6, 3.8), dpi=130)
            colors = [EMOTION_META[e]['color'] for e in EMOTIONS]
            bars = ax.barh([e.capitalize() for e in EMOTIONS], probs, color=colors, height=0.6)
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("Probability", fontsize=9, fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.5)

            for bar in bars:
                w = bar.get_width()
                if w > 0.03:
                    ax.annotate(f"{w:.1%}",
                                (w, bar.get_y() + bar.get_height() / 2),
                                xytext=(5, 0), textcoords="offset points",
                                va='center', ha='left', fontsize=8, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

        # 3. Attention Mechanism Visualization
        if attention_weights is not None and tokens is not None:
            st.markdown("---")
            st.subheader("3. Attention Mechanism Interpretation")
            st.write("The Bahdanau attention layer assigns dynamic weights to each token, revealing which words drove the prediction:")

            # Word highlight display
            html_content = get_highlighted_html(tokens, attention_weights, pred_label=pred_emotion, confidence=confidence)
            st.markdown(html_content, unsafe_allow_html=True)

            # Attention Heatmap plot
            st.write("")
            fig_attn = plot_attention_heatmap(tokens, attention_weights, pred_label=pred_emotion, confidence=confidence)
            st.pyplot(fig_attn)

            # Top keywords
            token_pairs = [(t, w) for t, w in zip(tokens, attention_weights) if t not in ['<pad>', '<unk>']]
            top_words = sorted(token_pairs, key=lambda x: -x[1])[:3]
            if top_words:
                kw_str = ", ".join([f"**{w}** ({score:.1%})" for w, score in top_words])
                st.info(f"🔑 **Top Key Driver Tokens:** {kw_str}")

# Benchmark & Analysis Section
st.markdown("---")
with st.expander("📊 View All Architecture Evaluation & Confusion Matrices"):
    if os.path.exists("results/model_comparison.png"):
        st.image("results/model_comparison.png", caption="Architecture Benchmark Comparison", use_container_width=True)

    cm_cols = st.columns(2)
    models_to_check = [
        ("BiLSTM + Attention", "results/confusion_matrix_bilstm___attention.png"),
        ("Fine-Tuned DistilBERT", "results/confusion_matrix_fine_tuned_distilbert.png"),
        ("LSTM", "results/confusion_matrix_lstm.png"),
        ("GRU", "results/confusion_matrix_gru.png")
    ]
    for i, (name, path) in enumerate(models_to_check):
        if os.path.exists(path):
            with cm_cols[i % 2]:
                st.image(path, caption=f"Confusion Matrix: {name}", use_container_width=True)

