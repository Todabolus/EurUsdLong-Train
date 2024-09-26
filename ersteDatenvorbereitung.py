import pandas as pd
import talib  # Bibliothek für technische Analyse-Indikatoren

def load_data(file_path):
    # Lesen der Datei ohne die Umwandlung von Datum und Zeit
    data = pd.read_csv(
        file_path, 
        sep=';', 
        encoding='utf-16le',  # Encoding auf UTF-16 Little Endian setzen
        decimal='.'
    )
    
    # Kombinieren von 'Date' und 'Time' Spalten und Konvertierung in Datetime-Objekt
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%Y.%m.%d %H:%M')
    data.set_index('Datetime', inplace=True)
    
    # Löschen der ursprünglichen 'Date' und 'Time' Spalten
    data.drop(columns=['Date', 'Time'], inplace=True)
    
    # Zerlegen der 'Datetime'-Spalte in verschiedene Komponenten
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['Hour'] = data.index.hour
    data['Minute'] = data.index.minute
    data['DayofWeek'] = data.index.dayofweek
    
    # Löschen der 'Datetime'-Spalte
    data.reset_index(drop=True, inplace=True)
    
    # Umordnen der Spalten: neue Spalten nach vorne verschieben
    cols = ['Month', 'Day', 'Hour', 'Minute', 'DayofWeek'] + [col for col in data.columns if col not in ['Month', 'Day', 'Hour', 'Minute', 'DayofWeek']]
    data = data[cols]
    
    return data

def calculate_technical_indicators(data):
    # RSI
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    
    # MACD
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Bollinger Bands
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    
    # ATR
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    
    # OBV
    data['OBV'] = talib.OBV(data['Close'], data['Volume'])
    
    # ROC
    data['ROC'] = talib.ROC(data['Close'], timeperiod=10)
    
    # Williams %R
    data['WilliamsR'] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)
    
    # Stochastic Oscillator
    data['Stoch_k'], data['Stoch_d'] = talib.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    
    # CCI
    data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
    
    # Berechnung der Ichimoku-Komponenten
    data['Ichimoku_9'] = (data['High'].rolling(window=9).max() + data['Low'].rolling(window=9).min()) / 2
    data['Ichimoku_26'] = (data['High'].rolling(window=26).max() + data['Low'].rolling(window=26).min()) / 2
    data['Ichimoku_span_A'] = ((data['Ichimoku_9'] + data['Ichimoku_26']) / 2).shift(26)
    data['Ichimoku_span_B'] = ((data['High'].rolling(window=52).max() + data['Low'].rolling(window=52).min()) / 2).shift(26)
    
    # CMF
    data['CMF'] = talib.ADOSC(data['High'], data['Low'], data['Close'], data['Volume'], fastperiod=3, slowperiod=10)
    
    # Keltner Channel (Middle Line)
    data['Keltner_middle'] = talib.MA(data['Close'], timeperiod=20, matype=talib.MA_Type.EMA)
    
    # Donchian Channel
    data['Donchian_upper'] = data['High'].rolling(window=20).max()
    data['Donchian_lower'] = data['Low'].rolling(window=20).min()
    
    # Vortex Indicator
    data['Vortex_pos'], data['Vortex_neg'] = talib.PLUS_DM(data['High'], data['Low'], timeperiod=14), talib.MINUS_DM(data['High'], data['Low'], timeperiod=14)
    
    # Parabolic SAR
    data['Parabolic_SAR'] = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
    
    # Momentum Indicator
    data['Momentum'] = talib.MOM(data['Close'], timeperiod=10)
    
    # VWAP (Approximation using typical price)
    data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (data['Typical_Price'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    # Chande Momentum Oscillator (CMO)
    data['CMO'] = talib.CMO(data['Close'], timeperiod=14)
    
    # Force Index
    data['Force_Index'] = data['Close'].diff() * data['Volume']

    return data

if __name__ == "__main__":
    data = load_data("C:/Users/G MAN/AppData/Roaming/MetaQuotes/Tester/D0E8209F77C8CF37AD8BF550E51FF075/Agent-127.0.0.1-3000/MQL5/Files/Data.csv")
    data = calculate_technical_indicators(data)
    
    # Speichern des vollständigen Datensatzes
    data.to_csv('full_data.csv', index=False)
    
    # Ausgabe der Anzahl der Zeilen im Datensatz
    print(f"Anzahl der Zeilen im Datensatz: {len(data)}")
