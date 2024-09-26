import os
import pandas as pd

def create_folders(folders):
    for folder in folders:
        subfolders = ["0", "1"]
        for subfolder in subfolders:
            path = os.path.join(folder, subfolder)
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Unterordner '{path}' wurde erstellt.")
            else:
                print(f"Unterordner '{path}' existiert bereits.")

def load_and_split_csv(file_path):
    # Lade den Datensatz
    df = pd.read_csv(file_path)
    
    segments = []  # Liste zur Speicherung der Segmente
    segment_size = 60  # Die Größe jedes Segments (inklusive der Zeile mit Minute 39)
    
    # Suche alle Zeilen, wo der Minutenwert 59 ist
    minute_39_indices = df.index[df['Minute'] == 1].tolist()
    
    # Für jeden Index in der Liste der gefundenen Minute == 59
    for index in minute_39_indices:
        # Prüfe, ob es genügend Zeilen vor dem aktuellen Index gibt (999 Zeilen davor + die aktuelle Zeile)
        if index >= segment_size - 1:
            # Extrahiere das Segment: 999 Zeilen vor dem aktuellen Index und die Zeile mit Minute 39
            segment = df.iloc[index - segment_size + 1:index + 1]
            segments.append(segment)
        else:
            # Wenn nicht genügend Zeilen vorhanden sind, wird das Segment verworfen
            print(f"Segment an Index {index} verworfen, nicht genug Zeilen vorhanden.")
    
    return segments


def classify_segment(segment, full_data, ambiguous_counter):
    # 1. Holen der letzten Zeile des Segments
    last_row = segment.iloc[-1]
    
    # 2. Speichern des Close-Werts als refClose
    refClose = last_row['Close']
    
    # 3. Ermitteln der referenzierten Zeile in full_data
    criteria = (
        (full_data['Month'] == last_row['Month']) &
        (full_data['Day'] == last_row['Day']) &
        (full_data['Hour'] == last_row['Hour']) &
        (full_data['Minute'] == last_row['Minute']) &
        (full_data['DayofWeek'] == last_row['DayofWeek']) &
        (full_data['Open'] == last_row['Open']) &
        (full_data['High'] == last_row['High']) &
        (full_data['Low'] == last_row['Low']) &
        (full_data['Close'] == last_row['Close'])
    )
    
    matching_indices = full_data.index[criteria].tolist()
    
    # 4. Wenn mehr als ein passender Index gefunden wird, verwerfen
    if len(matching_indices) != 1:
        ambiguous_counter[0] += 1  # Zählt die Ambiguität
        return None
    
    # 5. Wenn genau ein Index gefunden wurde
    ref_index = matching_indices[0]
    
    # 6. Gehe ab der referenzierten Zeile weiter durch full_data
    for i in range(ref_index + 1, len(full_data)):
        current_row = full_data.iloc[i]
        
        # Prüfe Bedingungen für High und Low
        if current_row['High'] > refClose + 0.001 and current_row['Low'] < refClose - 0.0002:
            return 0  # Beide Bedingungen am selben Index -> Zuweisung zu Ordner 0
        elif current_row['High'] > refClose + 0.001:
            return 1  # Ordner 1, da High Bedingung erfüllt
        elif current_row['Low'] < refClose - 0.001:
            return 0  # Ordner 0, da Low Bedingung erfüllt
    
    # Falls keine Bedingung erfüllt wurde, kehrt die Funktion None zurück
    return None
    

def save_segments(segments, full_data):
    total_segments = len(segments)
    train_end = int(total_segments * 0.7)
    val_end = int(total_segments * 0.85)

    counters = {
        "train_data": {"0": 1, "1": 1},
        "val_data": {"0": 1, "1": 1},
        "test_data": {"0": 1, "1": 1}
    }

    ambiguous_counter = [0]  # Zählt, wie viele Segmente mehr als einen Index gefunden haben

    for i, segment in enumerate(segments):
        label = classify_segment(segment, full_data, ambiguous_counter)
        if label is not None:
            if i < train_end:
                folder = "train_data"
            elif i < val_end:
                folder = "val_data"
            else:
                folder = "test_data"
            
            subfolder = str(label)
            filename = f"{folder}/{subfolder}/{counters[folder][subfolder]}.csv"
            segment.to_csv(filename, index=False)
            counters[folder][subfolder] += 1

    print(f"Anzahl der Segmente mit mehreren gefundenen Indizes: {ambiguous_counter[0]}")



if __name__ == "__main__":
    folders = ["train_data", "val_data", "test_data"]
    create_folders(folders)
    
    file_path = "full_data.csv"
    full_data = pd.read_csv(file_path)
    segments = load_and_split_csv(file_path)
    
    print(f"Gefundene Segmente: {len(segments)}")
    save_segments(segments, full_data)
    
    print("Daten erfolgreich aufgeteilt, klassifiziert und in den entsprechenden Ordnern gespeichert.")