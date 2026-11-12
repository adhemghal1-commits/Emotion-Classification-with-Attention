import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from src.dataset import (
    EMOTIONS,
    EMOTION_TO_ID,
    ID_TO_EMOTION,
    prepare_goemotions_df,
    Vocabulary,
    load_glove_matrix,
    EmotionDataset
)
from src.models import LSTMClassifier, GRUClassifier, BiLSTMAttentionClassifier
from src.train import train_model
from src.evaluate import evaluate_pytorch_model
from src.train_transformer import train_distilbert
from src.visualize import plot_model_comparisons, plot_attention_heatmap


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Load and explore GoEmotions dataset
    print("\n--- 1. Loading and Mapping GoEmotions Dataset to 6 Core Emotions ---")
    train_df = prepare_goemotions_df("Simplified/train-00000-of-00001.parquet", use_ekman=True)
    val_df = prepare_goemotions_df("Simplified/validation-00000-of-00001.parquet", use_ekman=True)
    test_df = prepare_goemotions_df("Simplified/test-00000-of-00001.parquet", use_ekman=True)

    print(f"Train samples:      {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples:       {len(test_df)}")
    print("\nClass distribution in Train:")
    print(train_df['emotion'].value_counts())

    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    # Compute class weights to handle imbalance
    classes = np.arange(len(EMOTIONS))
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_df['label'].values)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print("\nBalanced Class Weights:", {EMOTIONS[i]: round(weights[i], 3) for i in range(len(EMOTIONS))})

    # 2. Build Vocabulary
    print("\n--- 2. Building Vocabulary ---")
    vocab = Vocabulary()
    vocab.build_vocab(train_df['clean_text'].tolist(), min_freq=2)
    vocab_path = "saved_models/vocab.json"
    vocab.save(vocab_path)
    print(f"Vocabulary size: {len(vocab)} words. Saved to {vocab_path}")

    # 3. Load GloVe Embeddings
    print("\n--- 3. Initializing Pre-trained GloVe (100d) Embedding Layer ---")
    embedding_matrix = load_glove_matrix(vocab, embedding_dim=100)

    # Create PyTorch DataLoaders
    batch_size = 64
    max_len = 50
    train_dataset = EmotionDataset(train_df['clean_text'], train_df['label'], vocab, max_len=max_len)
    val_dataset = EmotionDataset(val_df['clean_text'], val_df['label'], vocab, max_len=max_len)
    test_dataset = EmotionDataset(test_df['clean_text'], test_df['label'], vocab, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

    results_list = []

    # 4. Train Model 1: LSTM
    print("\n--- 4. Training Architecture 1: LSTM ---")
    lstm_model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        num_layers=2,
        num_classes=len(EMOTIONS),
        dropout=0.3,
        weights_matrix=embedding_matrix,
        freeze_embedding=False
    )
    train_model(
        lstm_model,
        train_loader,
        val_loader,
        num_epochs=4,
        lr=1e-3,
        device=device,
        save_path="saved_models/lstm.pt",
        class_weights=class_weights_tensor
    )
    lstm_metrics = evaluate_pytorch_model(lstm_model, test_loader, device, model_name="LSTM", save_dir="results")
    results_list.append({
        'Model': 'LSTM',
        'Accuracy': lstm_metrics['Accuracy'],
        'Macro F1': lstm_metrics['Macro F1'],
        'Weighted F1': lstm_metrics['Weighted F1']
    })

    # 5. Train Model 2: GRU
    print("\n--- 5. Training Architecture 2: GRU ---")
    gru_model = GRUClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        num_layers=2,
        num_classes=len(EMOTIONS),
        dropout=0.3,
        weights_matrix=embedding_matrix,
        freeze_embedding=False
    )
    train_model(
        gru_model,
        train_loader,
        val_loader,
        num_epochs=4,
        lr=1e-3,
        device=device,
        save_path="saved_models/gru.pt",
        class_weights=class_weights_tensor
    )
    gru_metrics = evaluate_pytorch_model(gru_model, test_loader, device, model_name="GRU", save_dir="results")
    results_list.append({
        'Model': 'GRU',
        'Accuracy': gru_metrics['Accuracy'],
        'Macro F1': gru_metrics['Macro F1'],
        'Weighted F1': gru_metrics['Weighted F1']
    })

    # 6. Train Model 3: BiLSTM + Attention
    print("\n--- 6. Training Architecture 3: BiLSTM + Attention ---")
    bilstm_attn_model = BiLSTMAttentionClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        num_layers=2,
        num_classes=len(EMOTIONS),
        attention_dim=128,
        dropout=0.3,
        weights_matrix=embedding_matrix,
        freeze_embedding=False
    )
    train_model(
        bilstm_attn_model,
        train_loader,
        val_loader,
        num_epochs=4,
        lr=1e-3,
        device=device,
        save_path="saved_models/bilstm_attention.pt",
        class_weights=class_weights_tensor
    )
    bilstm_attn_metrics = evaluate_pytorch_model(
        bilstm_attn_model,
        test_loader,
        device,
        model_name="BiLSTM + Attention",
        save_dir="results"
    )
    results_list.append({
        'Model': 'BiLSTM + Attention',
        'Accuracy': bilstm_attn_metrics['Accuracy'],
        'Macro F1': bilstm_attn_metrics['Macro F1'],
        'Weighted F1': bilstm_attn_metrics['Weighted F1']
    })

    # 7. Train Model 4: Fine-Tuned Transformer (DistilBERT)
    print("\n--- 7. Training Architecture 4: Fine-Tuned Transformer (DistilBERT) ---")
    distilbert_metrics, _, _ = train_distilbert(
        train_df,
        val_df,
        test_df,
        model_name="distilbert-base-uncased",
        output_dir="saved_models/distilbert_emotion",
        num_epochs=2,
        batch_size=32,
        max_train_samples=4800
    )
    results_list.append({
        'Model': 'Fine-Tuned DistilBERT',
        'Accuracy': distilbert_metrics['Accuracy'],
        'Macro F1': distilbert_metrics['Macro F1'],
        'Weighted F1': distilbert_metrics['Weighted F1']
    })

    # 8. Model Comparison
    print("\n==================== Model Comparison Summary ====================")
    comparison_df = pd.DataFrame(results_list)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv("results/model_comparison.csv", index=False)
    plot_model_comparisons(comparison_df, save_path="results/model_comparison.png")
    print("Comparison table saved to results/model_comparison.csv and plot to results/model_comparison.png")

    # 9. Plot Attention Heatmaps for Interpretation
    print("\n--- 9. Generating Attention Heatmaps for Interpretation ---")
    sample_texts = [
        ("I am absolutely thrilled and overjoyed with this wonderful news!", "joy"),
        ("I feel completely devastated, empty, and heart-broken today.", "sadness"),
        ("This is infuriating and ridiculous, I am so furious right now!", "anger"),
        ("I am terrified of what might happen next, it gives me chills.", "fear"),
        ("Wait what? I was completely stunned and never expected that!", "surprise"),
        ("That rancid rotten smell is absolutely sickening and repulsive.", "disgust")
    ]

    bilstm_attn_model.eval()
    for text, true_emo in sample_texts:
        clean = text.lower()
        ids, length, tokens = vocab.encode(clean, max_len=max_len)
        inp = torch.tensor([ids], dtype=torch.long).to(device)
        with torch.no_grad():
            logits, weights = bilstm_attn_model(inp, return_attention=True)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_id = int(np.argmax(probs))
            pred_emo = ID_TO_EMOTION[pred_id]
            conf = float(probs[pred_id])
            token_weights = weights[0].cpu().numpy()[:length]

        save_fig_path = f"results/attention_{true_emo}.png"
        plot_attention_heatmap(
            tokens=tokens,
            attention_weights=token_weights,
            true_label=true_emo,
            pred_label=pred_emo,
            confidence=conf,
            save_path=save_fig_path
        )
        print(f"Sample [{true_emo}] -> Predicted: {pred_emo} ({conf:.1%}) | Saved heatmap to {save_fig_path}")

    print("\n--- All 9 Stages Finished Successfully! ---")


if __name__ == "__main__":
    main()

