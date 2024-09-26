import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, GRU, Conv1D, MaxPooling1D, Add, Input, Lambda, TimeDistributed, Bidirectional, Attention, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, TensorBoard
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop
import keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

def load_data():
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_val = np.load('X_val.npy')
    y_val = np.load('y_val.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(32, input_shape=input_shape, return_sequences=True, kernel_regularizer=l2(0.02))))
    model.add(Bidirectional(LSTM(16, kernel_regularizer=l2(0.02))))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Erhöhte Dropout-Rate
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def main():
    print("Lade vorbereitete Daten...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print("Erstelle Modell...")
    model = create_model(X_train.shape[1:])

    print("Starte Training...")
    tensorboard_callback = TensorBoard(log_dir="./logs")
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    start_time = time.time()
    
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=150,
                        batch_size=32,
                        callbacks=[tensorboard_callback, early_stopping_callback])

    training_time = time.time() - start_time
    print(f"Training abgeschlossen in {training_time:.2f} Sekunden.")
    
    print("Evaluierung des Modells...")
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)

    print("Speichern des Modells...")
    model.save('model.h5')

    print("Modelltraining und Evaluierung abgeschlossen.")

if __name__ == "__main__":
    main()