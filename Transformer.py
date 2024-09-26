import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE  # SMOTE für Oversampling
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

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

# Transformer-basierte Modellarchitektur
def build_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Transformer Encoder Block
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.1)(x, x)
    x = layers.Dropout(0.1)(x)
    x = layers.Add()([x, inputs])  # Residual connection

    # Weitere Transformer-Blöcke können hier hinzugefügt werden
    x = layers.GlobalAveragePooling1D()(x)

    # Klassifikations-Head
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', 'Precision'])
    
    return model

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

    # SMOTE auf den Trainingsdaten anwenden, um Klasse 1 zu oversamplen
    smote = SMOTE(random_state=42)
    train_sequences_scaled_resampled, train_labels_resampled = smote.fit_resample(train_sequences_scaled.reshape(train_sequences_scaled.shape[0], -1), train_labels)

    # Reshape nach dem Oversampling
    train_sequences_scaled_resampled = train_sequences_scaled_resampled.reshape(-1, train_sequences_scaled.shape[1], train_sequences_scaled.shape[2])

    # Berechne class weights für das Ungleichgewicht
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_dict = dict(enumerate(class_weights))

    # Transformer-Modell erstellen
    input_shape = train_sequences_scaled_resampled.shape[1:]
    model = build_transformer_model(input_shape)

    # Early Stopping Callback hinzufügen
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Modell trainieren
    history = model.fit(train_sequences_scaled_resampled, train_labels_resampled, 
                        validation_data=(val_sequences_scaled, val_labels),
                        epochs=100, batch_size=32, class_weight=class_weights_dict,
                        callbacks=[early_stopping])

    # Vorhersagen mit Wahrscheinlichkeiten
    test_probabilities = model.predict(test_sequences_scaled)
    
    # Threshold auf 0.6 setzen, um weniger False Positives zu haben
    threshold = 0.7
    test_predictions = (test_probabilities >= threshold).astype(int)

    # Modellbewertung
    conf_matrix = confusion_matrix(test_labels, test_predictions)

    # Plot Confusion Matrix
    plot_confusion_matrix(conf_matrix, class_names=['0', '1'])

    # Classification report für detailliertere Metriken
    print(classification_report(test_labels, test_predictions, target_names=['0', '1']))

# Ruft die main-Methode auf
if __name__ == "__main__":
    main()
