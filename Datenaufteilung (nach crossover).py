import os
import pandas as pd

# 1. Erstellen der Ordnerstruktur
folders = ['train_data', 'val_data', 'test_data']
subfolders = ['0', '1']

for folder in folders:
    for subfolder in subfolders:
        path = os.path.join(folder, subfolder)
        os.makedirs(path, exist_ok=True)

print("Ordnerstruktur erstellt.")

# 2. CSV-Datei einlesen und Crossover-Indizes sammeln
df = pd.read_csv('full_data.csv')
crossover_indices = df.index[(df['Crossover'] == True) & (df.index >= 1500)].tolist()
total_crossover = len(crossover_indices)

print(f"Es wurden {total_crossover} Crossover gefunden (ab Index 1500).")

# 3. Aufteilen in train, val und test
train_end = int(total_crossover * 0.7)
val_end = train_end + int(total_crossover * 0.15)

train_indices = crossover_indices[:train_end]
val_indices = crossover_indices[train_end:val_end]
test_indices = crossover_indices[val_end:]

print(f"Aufteilung: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test.")

# 4. Label-Zuweisung und CSV-Dateien speichern
def get_label(crossover_price, close_prices):
    for price in close_prices:
        if price >= crossover_price + 20:
            return 1
        if price <= crossover_price - 2:
            return 0
    return None

def save_data(indices, folder_name):
    counter_0 = 1
    counter_1 = 1
    for index in indices:
        crossover_price = df.iloc[index]['Close']
        close_prices = df.iloc[index: index + 1500]['Close'].tolist()

        label = get_label(crossover_price, close_prices)

        if label is not None:
            subfolder = str(label)
            data_to_save = df.iloc[index-1499:index+1]  # 1500 Zeilen rückwirkend
            if label == 1:
                file_path = os.path.join(folder_name, subfolder, f"{counter_1}.csv")
                counter_1 += 1
            else:
                file_path = os.path.join(folder_name, subfolder, f"{counter_0}.csv")
                counter_0 += 1

            data_to_save.to_csv(file_path, index=False)
            print(f"Datei {file_path} gespeichert.")

print("Speichern von Trainingsdaten...")
save_data(train_indices, 'train_data')

print("Speichern von Validierungsdaten...")
save_data(val_indices, 'val_data')

print("Speichern von Testdaten...")
save_data(test_indices, 'test_data')

print("Alle Daten wurden erfolgreich gespeichert.")