# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from sklearn.metrics import confusion_matrix
import sys
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.words_model.model import KannadaSignModel

def load_classes(class_file='class_indices.json'):
    with open(class_file, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)

    # Convert string keys to int keys
    idx_to_class = {int(k): v for k, v in class_indices.items()}
    class_to_idx = {v: int(k) for k, v in class_indices.items()}
    return idx_to_class, class_to_idx

def load_model(model_path, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = KannadaSignModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device

def preprocess_data(data_path):
    data = np.load(data_path)

    # Debug shape
    original_shape = data.shape
    print(f"Original shape: {original_shape}")

    # Reshape to (1, 126, frames)
    data = data.reshape(1, -1, data.shape[0])
    print(f"Reshaped to: {data.shape}")

    return torch.FloatTensor(data)

def predict_class(model, data, device, idx_to_class):
    model.eval()
    data = data.to(device)

    # Fix shape mismatch: (batch, seq_len, features) -> (batch, features, seq_len)
    data = data.permute(0, 2, 1)

    with torch.no_grad():
        outputs = model(data)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    print(f"Predicted index: {predicted.item()}, Available keys: {list(idx_to_class.keys())}")

    predicted_class = idx_to_class[predicted.item()]
    return predicted_class, confidence.item()

def analyze_confusion(val_dir, model, device, idx_to_class, class_to_idx):
    all_true_labels = []
    all_predictions = []
    all_confidences = []

    for class_name in os.listdir(val_dir):
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"Analyzing class '{class_name}'...")

        for sample_file in os.listdir(class_dir):
            if not sample_file.endswith('.npy'):
                continue

            sample_path = os.path.join(class_dir, sample_file)

            # Preprocess the data
            data = preprocess_data(sample_path)

            # Predict the class
            predicted_class, confidence = predict_class(model, data, device, idx_to_class)

            all_true_labels.append(class_name)
            all_predictions.append(predicted_class)
            all_confidences.append(confidence)

            print(f"  {'✓' if predicted_class == class_name else '✗'} {sample_file}: Predicted {predicted_class} ({confidence:.2f})")

    return all_true_labels, all_predictions, all_confidences

def plot_confusion_matrix(true_labels, predictions, idx_to_class):
    classes = list(idx_to_class.values())

    cm = confusion_matrix(true_labels, predictions, labels=classes)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cm_df = pd.DataFrame(cm_normalized, index=classes, columns=classes)

    confused_pairs = []
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((true_class, pred_class, cm[i, j], cm_normalized[i, j]))

    confused_pairs.sort(key=lambda x: x[3], reverse=True)

    print("\n=== Most Confused Class Pairs ===")
    for true_class, pred_class, count, norm_value in confused_pairs[:15]:
        print(f"True: {true_class}, Predicted: {pred_class} - Count: {count} ({norm_value:.2%})")

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_df, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('detailed_confusion_matrix.png', dpi=300)
    plt.close()

    # Plot top confused pairs
    top_confused = confused_pairs[:10]
    if top_confused:
        plt.figure(figsize=(12, 8))
        pairs = [f"{t} → {p}" for t, p, _, _ in top_confused]
        values = [c for _, _, c, _ in top_confused]

        plt.barh(pairs, values, color='skyblue')
        plt.xlabel('Number of Confusions')
        plt.title('Top 10 Most Confused Class Pairs')
        plt.tight_layout()
        plt.savefig('top_confused_pairs.png', dpi=300)
        plt.close()

def analyze_confidence_distribution(true_labels, predictions, confidences, idx_to_class):
    df = pd.DataFrame({
        'True': true_labels,
        'Predicted': predictions,
        'Confidence': confidences,
        'Correct': [t == p for t, p in zip(true_labels, predictions)]
    })

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Confidence', hue='Correct', bins=20, kde=True)
    plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
    plt.savefig('confidence_distribution.png', dpi=300)
    plt.close()

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='True', y='Confidence', hue='Correct')
    plt.xticks(rotation=90)
    plt.title('Confidence by Class')
    plt.tight_layout()
    plt.savefig('confidence_by_class.png', dpi=300)
    plt.close()

    class_confidence = df.groupby('True')['Confidence'].mean().sort_values()
    print("\n=== Average Confidence by Class ===")
    for class_name, avg_conf in class_confidence.items():
        print(f"{class_name}: {avg_conf:.3f}")

def main():
    # Load classes
    idx_to_class, class_to_idx = load_classes()
    num_classes = len(idx_to_class)

    # Load model
    model_path = 'model/checkpoints/best_model.pth'
    model, device = load_model(model_path, num_classes)

    val_dir = 'dataset/new_processed_video/val'

    true_labels, predictions, confidences = analyze_confusion(val_dir, model, device, idx_to_class, class_to_idx)

    plot_confusion_matrix(true_labels, predictions, idx_to_class)

    analyze_confidence_distribution(true_labels, predictions, confidences, idx_to_class)

if __name__ == "__main__":
    main()
