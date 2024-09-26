import MetaTrader5 as mt5
import csv
import os
from datetime import datetime, timedelta

# MetaTrader 5 initialisieren
if not mt5.initialize():
    print("MetaTrader 5 konnte nicht initialisiert werden")
    mt5.shutdown()

# MT5 account credentials
login = 13072047  # Replace with your account number
password = "epoPH17##"  # Replace with your account password
server = "FundedNext-Server 2"  # Replace with your broker's server name

if not mt5.login(login, password, server):
    print(f"Login fehlgeschlagen, Fehlercode: {mt5.last_error()}")
    mt5.shutdown()

# Dateipfad und Dateiname
file_path = "C:/Daten/data.csv"

# CSV-Datei erstellen, falls sie noch nicht existiert
if not os.path.exists(file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volumen'])

# Funktion zum Abrufen und Speichern historischer Kerzendaten
def save_historical_data(symbol, timeframe, start_date, end_date):
    while start_date < end_date:
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, start_date + timedelta(days=1))
        
        if rates is not None and len(rates) > 0:
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                for rate in rates:
                    date_time = datetime.fromtimestamp(rate['time'])
                    date = date_time.strftime('%d.%m.%Y')
                    time_str = date_time.strftime('%H:%M:%S')
                    open_price = f"{rate['open']:.5f}".replace('.', ',')
                    high_price = f"{rate['high']:.5f}".replace('.', ',')
                    low_price = f"{rate['low']:.5f}".replace('.', ',')
                    close_price = f"{rate['close']:.5f}".replace('.', ',')
                    volume = rate['tick_volume']
                    writer.writerow([date, time_str, open_price, high_price, low_price, close_price, volume])
            print(f"Gespeicherte Daten für {start_date.strftime('%d.%m.%Y')}")

        start_date += timedelta(days=1)

# Zeitraum für historische Daten (ab 1972)
start_date = datetime(1972, 1, 1)
end_date = datetime.now()

# Historische Daten speichern
save_historical_data("EURUSD", mt5.TIMEFRAME_M1, start_date, end_date)

# MetaTrader 5 Sitzung beenden
mt5.shutdown()
