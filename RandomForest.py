import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Funktion, um den besten Schwellenwert anhand der Precision-Recall-Kurve zu finden
def find_best_threshold(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_idx = np.argmax(f1_scores)  # Finde den Index mit dem höchsten F1-Score
    best_threshold = thresholds[best_idx]
    print(f"Best threshold based on F1-Score: {best_threshold:.2f}")
    return best_threshold

# 1. Daten laden
def load_data_from_folders(base_dir):
    sequences = []
    labels = []
    
    for label in ['0', '1']:
        folder_path = os.path.join(base_dir, label)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                data = pd.read_csv(file_path)
                sequences.append(data.values)
                labels.append(int(label))
    
    return np.array(sequences), np.array(labels)

# Plot der Konfusionsmatrix
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title="Confusion Matrix",
           ylabel='True label',
           xlabel='Predicted label')

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

# Funktion für Early Stopping
def early_stopping(model, X_train, y_train, X_val, y_val, patience=20):
    best_model = None
    best_score = -np.inf
    epochs_no_improve = 0
    train_accs = []
    val_accs = []

    for i in range(1, model.n_estimators + 1):
        model.n_estimators = i
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        train_accs.append(train_score)
        val_accs.append(val_score)
        print(f"Epoch {i}, Train Accuracy: {train_score}, Validation Accuracy: {val_score}")
        
        if val_score > best_score:
            best_score = val_score
            best_model = model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {i}")
            break

    # Plot Training and Validation Accuracy over Epochs
    plt.figure()
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Accuracy')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

    return best_model

# Funktion für ROC-Kurve
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Funktion für Precision-Recall-Kurve
def plot_precision_recall_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

# Hauptfunktion zum Trainieren und Evaluieren
def main():
    # Lade Daten aus den Ordnern
    train_sequences, train_labels = load_data_from_folders('train_data')
    val_sequences, val_labels = load_data_from_folders('val_data')
    test_sequences, test_labels = load_data_from_folders('test_data')

    # 2. NaN-Werte durch Mittelwert ersetzen
    def fill_nan_with_mean(sequences):
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        sequences_df = pd.DataFrame(sequences_reshaped)
        sequences_df.fillna(sequences_df.mean(), inplace=True)
        return sequences_df.to_numpy().reshape(sequences.shape)

    train_sequences_filled = fill_nan_with_mean(train_sequences)
    val_sequences_filled = fill_nan_with_mean(val_sequences)
    test_sequences_filled = fill_nan_with_mean(test_sequences)

    # Daten skalieren
    scaler = StandardScaler()
    train_sequences_reshaped = train_sequences_filled.reshape(-1, train_sequences_filled.shape[-1])
    val_sequences_reshaped = val_sequences_filled.reshape(-1, val_sequences_filled.shape[-1])
    test_sequences_reshaped = test_sequences_filled.reshape(-1, test_sequences_filled.shape[-1])

    train_sequences_scaled = scaler.fit_transform(train_sequences_reshaped).reshape(train_sequences_filled.shape)
    val_sequences_scaled = scaler.transform(val_sequences_reshaped).reshape(val_sequences_filled.shape)
    test_sequences_scaled = scaler.transform(test_sequences_reshaped).reshape(test_sequences_filled.shape)

    # Daten flach machen für Random Forest
    train_sequences_flat = train_sequences_scaled.reshape(train_sequences_scaled.shape[0], -1)
    val_sequences_flat = val_sequences_scaled.reshape(val_sequences_scaled.shape[0], -1)
    test_sequences_flat = test_sequences_scaled.reshape(test_sequences_scaled.shape[0], -1)

    # SMOTE auf den Trainingsdaten anwenden, um Klasse 1 zu oversamplen
    smote = SMOTE(random_state=42)
    train_sequences_flat_resampled, train_labels_resampled = smote.fit_resample(train_sequences_flat, train_labels)

    # Berechne class weights für das Ungleichgewicht
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_dict = dict(enumerate(class_weights))

    # Random Forest Modell erstellen
    model = RandomForestClassifier(
        n_estimators=2000,  # Maximal 2000 Bäume
        max_depth=20,  # Fester Wert für max_depth
        min_samples_split=5,  # Fester Wert für min_samples_split
        min_samples_leaf=1,  # Fester Wert für min_samples_leaf
        class_weight=class_weights_dict,
        random_state=42,
        warm_start=True  # Ermöglicht inkrementelles Hinzufügen von Bäumen
    )

    # Early Stopping anwenden
    best_model = early_stopping(model, train_sequences_flat_resampled, train_labels_resampled, val_sequences_flat, val_labels)

    # Vorhersagen mit Wahrscheinlichkeiten
    test_probabilities = best_model.predict_proba(test_sequences_flat)[:, 1]

    # Finde den besten Schwellenwert basierend auf der Precision-Recall-Kurve
    best_threshold = find_best_threshold(test_labels, test_probabilities)

    # Verwende den neuen Schwellenwert, um Vorhersagen zu treffen
    test_predictions = (test_probabilities >= best_threshold).astype(int)

    # Modellbewertung
    conf_matrix = confusion_matrix(test_labels, test_predictions)

    # Plot Confusion Matrix
    plot_confusion_matrix(conf_matrix, class_names=['0', '1'])

    # Classification report für detailliertere Metriken
    print(classification_report(test_labels, test_predictions, target_names=['0', '1']))

    # ROC-Kurve plotten
    plot_roc_curve(test_labels, test_probabilities)

    # Precision-Recall-Kurve plotten
    plot_precision_recall_curve(test_labels, test_probabilities)

# Ruft die main-Methode auf
if __name__ == "__main__":
    main()
