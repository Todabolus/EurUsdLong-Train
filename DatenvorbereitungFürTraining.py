import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess_data(base_path):
    data = []
    labels = []
    
    for label in [0, 1]:
        folder = os.path.join(base_path, str(label))
        files = os.listdir(folder)
        
        for file in files:
            file_path = os.path.join(folder, file)
            df = pd.read_csv(file_path)
            
            # Fehlende Werte auffüllen: nur eine Methode verwenden (hier forward fill)
            df = df.ffill()

            # Sicherstellen, dass alle Dateien die gleiche Form haben (hier als Beispiel: reshaping auf gleiche Anzahl von Features)
            if len(data) > 0:
                # Prüfen, ob die neue Datei die gleiche Anzahl von Spalten hat wie die vorherige
                if df.shape[1] != data[0].shape[1]:
                    print(f"Warnung: Inkonsistente Anzahl an Spalten in Datei {file}")
                    continue
            
            data.append(df.values)
            labels.append(label)

    # Alle Daten in ein 3D-Array konvertieren, falls sie unterschiedliche Längen haben
    max_length = max([x.shape[0] for x in data])
    data = np.array([np.pad(x, ((0, max_length - x.shape[0]), (0, 0)), mode='constant') for x in data])
    
    labels = np.array(labels)
    
    print(f"Geladene Datenform: {data.shape}")
    print(f"Geladene Labelsform: {labels.shape}")

    # Überprüfung auf NaN-Werte
    if np.isnan(data).any():
        print("Warnung: NaN-Werte in den Daten gefunden. Es werden 0-Werte eingefügt.")
        data = np.nan_to_num(data)

    return data, labels

def balance_classes(X, y):
    from sklearn.utils import resample
    
    X_0, X_1 = X[y == 0], X[y == 1]
    y_0, y_1 = y[y == 0], y[y == 1]
    
    # Hochsampeln der Klasse 1 auf die Größe der Klasse 0
    X_1_upsampled, y_1_upsampled = resample(X_1, y_1,
                                            replace=True,
                                            n_samples=len(X_0),
                                            random_state=42)
    
    # Klassen zusammenfügen
    X_balanced = np.concatenate((X_0, X_1_upsampled), axis=0)
    y_balanced = np.concatenate((y_0, y_1_upsampled), axis=0)
    
    print(f"Balancierte Datenform: {X_balanced.shape}")
    print(f"Balancierte Labelsform: {y_balanced.shape}")

    return X_balanced, y_balanced

def save_data(X_train, y_train, X_val, y_val, X_test, y_test):
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

def main():
    print("Datenvorbereitung beginnt...")

    train_data_path = 'train_data'
    val_data_path = 'val_data'
    test_data_path = 'test_data'

    # Daten laden
    X_train, y_train = load_and_preprocess_data(train_data_path)
    X_val, y_val = load_and_preprocess_data(val_data_path)
    X_test, y_test = load_and_preprocess_data(test_data_path)
    
    print("Daten geladen und fehlende Werte behandelt.")

    # Klassen balancieren
    X_train, y_train = balance_classes(X_train, y_train)
    X_val, y_val = balance_classes(X_val, y_val)
    
    print("Klassenbalancierung abgeschlossen.")

    # Daten skalieren (nachdem wir sicherstellen, dass die Daten 2D sind)
    scaler = StandardScaler()
    
    # Aufpassen: Wir transformieren nur über die Feature-Dimension
    X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten der Daten für die Skalierung
    X_train = scaler.fit_transform(X_train)
    
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_val = scaler.transform(X_val)

    X_test = X_test.reshape(X_test.shape[0], -1)
    X_test = scaler.transform(X_test)
    
    print("Normalisierung abgeschlossen.")

    # Scaler speichern
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler gespeichert.")

    # Daten speichern
    save_data(X_train, y_train, X_val, y_val, X_test, y_test)
    
    print("Daten gespeichert. Datenvorbereitung abgeschlossen.")

if __name__ == "__main__":
    main()
