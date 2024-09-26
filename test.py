import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Pfade zu Ihren Datenverzeichnissen
train_dir = 'train_data'
val_dir = 'val_data'
test_dir = 'test_data'

def load_data(data_dir):
    data = []
    labels = []
    for label in ['0', '1']:
        label_dir = os.path.join(data_dir, label)
        for filename in os.listdir(label_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(label_dir, filename)
                # CSV-Datei einlesen
                df = pd.read_csv(file_path)
                # Nullwerte behandeln
                df.ffill(inplace=True)  # Forward-fill
                df.bfill(inplace=True)  # Backward-fill
                df.fillna(0, inplace=True)  # Verbleibende NaNs mit Null füllen
                # In NumPy-Array konvertieren
                arr = df.to_numpy()
                data.append(arr)
                labels.append(int(label))
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

# Daten laden
print("Lade Trainingsdaten...")
X_train, y_train = load_data(train_dir)
print("Lade Validierungsdaten...")
X_val, y_val = load_data(val_dir)
print("Lade Testdaten...")
X_test, y_test = load_data(test_dir)

# Datenformen überprüfen
print("Trainingsdatenform:", X_train.shape)
print("Validierungsdatenform:", X_val.shape)
print("Testdatenform:", X_test.shape)

# Daten skalieren
scaler = StandardScaler()

# Daten für Skalierung in 2D umformen
num_samples_train, timesteps, num_features = X_train.shape
X_train_reshaped = X_train.reshape(-1, num_features)
X_val_reshaped = X_val.reshape(-1, num_features)
X_test_reshaped = X_test.reshape(-1, num_features)

# Skalierer auf Trainingsdaten fitten und transformieren
scaler.fit(X_train_reshaped)
X_train_scaled = scaler.transform(X_train_reshaped).reshape(num_samples_train, timesteps, num_features)
X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

# Modell erstellen
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(timesteps, num_features)))  # Erste LSTM-Schicht
model.add(Dropout(0.3))  # Dropout-Schicht zur Reduktion von Overfitting
model.add(LSTM(64))  # Zweite LSTM-Schicht
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Binäre Klassifikation

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Modell trainieren
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,  # Erhöhung der Epochenanzahl
    batch_size=64,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Modell auf Testdaten evaluieren
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print("Testgenauigkeit:", test_acc)

# Vorhersagen auf Testdaten generieren
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Konfusionsmatrix berechnen
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])

# Konfusionsmatrix plotten
disp.plot(cmap=plt.cm.Blues)
plt.title('Konfusionsmatrix auf Testdaten')
plt.show()

# Klassifikationsbericht
print("Klassifikationsbericht:")
print(classification_report(y_test, y_pred, target_names=['Klasse 0', 'Klasse 1']))

# Finales Modell speichern
model.save('final_model.h5')
