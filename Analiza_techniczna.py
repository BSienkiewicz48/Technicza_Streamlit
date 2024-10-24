import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import streamlit as st
import math
import numpy as np

# Funkcje do obliczania wskaźników technicznych

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, span_short=12, span_long=26, span_signal=9):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(series, window=20, std_dev=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return sma, upper, lower

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    lowest_low = low.rolling(window=k_window, min_periods=1).min()
    highest_high = high.rolling(window=k_window, min_periods=1).max()

    denominator = highest_high - lowest_low
    denominator = denominator.replace(0, 1)

    stochastic_k = 100 * ((close - lowest_low) / denominator)
    stochastic_d = stochastic_k.rolling(window=d_window, min_periods=1).mean()
    return stochastic_k, stochastic_d

def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close_prev = (high - close.shift()).abs()
    low_close_prev = (low - close.shift()).abs()

    tr_df = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
    tr = tr_df.max(axis=1)

    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def calculate_obv(close, volume):
    obv = (volume * ((close > close.shift()).astype(int) - (close < close.shift()).astype(int))).cumsum()
    return obv

# Funkcja do analizy akcji
def analyze_stock(ticker, indicators_weights):
    # Pobierz dane historyczne z dłuższego okresu (2 lata)
    data = yf.download(ticker, period='2y', interval='1d')
    if data.empty:
        st.warning(f"Brak danych dla {ticker}")
        return None, 0

    # Flatten MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns.values]

    # Check if required columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.warning(f"Dane dla {ticker} nie zawierają kolumn: {', '.join(missing_columns)}")
        return None, 0

    # Oblicz wskaźniki techniczne
    data['RSI'] = calculate_rsi(data['Close'], period=14)
    data['SMA50'] = data['Close'].rolling(window=50, min_periods=1).mean()
    data['SMA200'] = data['Close'].rolling(window=200, min_periods=1).mean()
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA100'] = data['Close'].ewm(span=100, adjust=False).mean()

    # MACD
    data['MACD'], data['Signal'] = calculate_macd(data['Close'], span_short=12, span_long=26, span_signal=9)

    # Bollinger Bands
    data['Bollinger_Middle'], data['Bollinger_High'], data['Bollinger_Low'] = calculate_bollinger_bands(data['Close'], window=20, std_dev=2)

    # Stochastic Oscillator
    data['Stochastic_K'], data['Stochastic_D'] = calculate_stochastic(data['High'], data['Low'], data['Close'], k_window=14, d_window=3)

    # ATR
    data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'], period=14)

    # On-Balance Volume
    data['OBV'] = calculate_obv(data['Close'], data['Volume'])

    # Najnowsze wartości
    latest_close = float(data['Close'].iloc[-1])
    latest_rsi = float(data['RSI'].iloc[-1])
    latest_sma50 = float(data['SMA50'].iloc[-1])
    latest_sma200 = float(data['SMA200'].iloc[-1])
    latest_ema20 = float(data['EMA20'].iloc[-1])
    latest_ema100 = float(data['EMA100'].iloc[-1])
    latest_macd = float(data['MACD'].iloc[-1])
    latest_signal = float(data['Signal'].iloc[-1])
    latest_boll_high = float(data['Bollinger_High'].iloc[-1])
    latest_boll_low = float(data['Bollinger_Low'].iloc[-1])
    latest_stoch_k = float(data['Stochastic_K'].iloc[-1])
    latest_stoch_d = float(data['Stochastic_D'].iloc[-1])
    latest_atr = float(data['ATR'].iloc[-1])
    latest_obv = float(data['OBV'].iloc[-1])

    # Inicjalizacja punktacji dla rekomendacji
    score = 0

    # RSI Analysis (Mniej Ważny)
    if latest_rsi < 30:
        score += indicators_weights['RSI']['buy']
    elif latest_rsi > 70:
        score += indicators_weights['RSI']['sell']
    else:
        score += indicators_weights['RSI']['neutral']

    # SMA Analysis (Ważny)
    if latest_close > latest_sma50 > latest_sma200:
        score += indicators_weights['SMA']['bullish']
    elif latest_close < latest_sma50 < latest_sma200:
        score += indicators_weights['SMA']['bearish']
    else:
        score += indicators_weights['SMA']['neutral']

    # EMA Analysis (Ważny)
    if latest_ema20 > latest_ema100:
        score += indicators_weights['EMA']['bullish']
    elif latest_ema20 < latest_ema100:
        score += indicators_weights['EMA']['bearish']
    else:
        score += indicators_weights['EMA']['neutral']

    # MACD Analysis (Ważny)
    if latest_macd > latest_signal:
        score += indicators_weights['MACD']['bullish']
    elif latest_macd < latest_signal:
        score += indicators_weights['MACD']['bearish']
    else:
        score += indicators_weights['MACD']['neutral']

    # Bollinger Bands Analysis (Mniej Ważny)
    if latest_close > latest_boll_high:
        score += indicators_weights['Bollinger']['sell']
    elif latest_close < latest_boll_low:
        score += indicators_weights['Bollinger']['buy']
    else:
        score += indicators_weights['Bollinger']['neutral']

    # Stochastic Oscillator Analysis (Mniej Ważny)
    if latest_stoch_k < 20 and latest_stoch_d < 20:
        score += indicators_weights['Stochastic']['buy']
    elif latest_stoch_k > 80 and latest_stoch_d > 80:
        score += indicators_weights['Stochastic']['sell']
    else:
        score += indicators_weights['Stochastic']['neutral']

    # ATR Analysis (Mniej Ważny)
    if latest_atr > data['ATR'].mean():
        score += indicators_weights['ATR']['high_volatility']
    else:
        score += indicators_weights['ATR']['low_volatility']

    # OBV Analysis (Mniej Ważny)
    recent_obv_mean = data['OBV'].iloc[-30:].mean()
    if latest_obv > recent_obv_mean:
        score += indicators_weights['OBV']['bullish']
    elif latest_obv < recent_obv_mean:
        score += indicators_weights['OBV']['bearish']
    else:
        score += indicators_weights['OBV']['neutral']

    # Upewnij się, że punktacja nie jest ujemna
    score = max(score, 0)

    # Zwracamy dane oraz wynik
    return data, score

# Funkcja do rysowania wykresu
def plot_stock(ticker, data, score, max_score):
    # Przygotowanie danych dla mplfinance
    df = data.copy()
    df.index.name = 'Date'

    # Upewnij się, że indeks jest typu DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Subset data to the last 6 months
    six_months_ago = df.index.max() - pd.DateOffset(months=6)
    df_plot = df.loc[df.index >= six_months_ago].copy()

    # Ujednolicenie kolumn, jeśli mają MultiIndex
    if isinstance(df_plot.columns, pd.MultiIndex):
        df_plot.columns = df_plot.columns.get_level_values(-1)

    # Upewnij się, że 'Open', 'High', 'Low', 'Close', 'Volume' są dostępne
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df_plot.columns]
    if missing_cols:
        st.error(f"Brakujące kolumny: {', '.join(missing_cols)} dla {ticker}")
        return None

    for col in required_cols:
        try:
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
        except Exception as e:
            st.error(f"Błąd konwersji kolumny {col} dla {ticker}: {e}")
            return None

    # Usuń wiersze z NaN w kluczowych kolumnach
    df_plot.dropna(subset=required_cols, inplace=True)

    # Upewnij się, że po usunięciu NaN nie pozostały puste dane
    if df_plot.empty:
        st.error(f"Brak danych po usunięciu NaN dla {ticker}")
        return None

    # Obliczenie minimalnej i maksymalnej wartości wstęg Bollingera
    if 'Bollinger_Low' not in df_plot.columns or 'Bollinger_High' not in df_plot.columns:
        st.error(f"Brak kolumn Bollinger Bands dla {ticker}")
        return None

    min_boll_low = df_plot['Bollinger_Low'].min()
    max_boll_high = df_plot['Bollinger_High'].max()

    # Definiowanie dodatkowych wskaźników do wykresu
    apds = []

    # EMA20 i EMA100
    if 'EMA20' in df_plot.columns:
        apds.append(mpf.make_addplot(df_plot['EMA20'], color='blue', width=1))
    if 'EMA100' in df_plot.columns:
        apds.append(mpf.make_addplot(df_plot['EMA100'], color='orange', width=1))

    # Bollinger Bands
    apds.append(mpf.make_addplot(df_plot['Bollinger_High'], color='red', linestyle='--'))
    apds.append(mpf.make_addplot(df_plot['Bollinger_Low'], color='green', linestyle='--'))

    # OBV Plot (panel=2)
    if 'OBV' in df_plot.columns:
        obv_plot = mpf.make_addplot(df_plot['OBV'], panel=2, color='purple', ylabel='OBV')
        apds.append(obv_plot)
    else:
        st.warning(f"Brak kolumny OBV dla {ticker}")

    # Rysowanie wykresu świecowego z dodatkowymi wskaźnikami
    try:
        fig, axes = mpf.plot(
            df_plot,
            type='candle',
            style='charles',
            title='',  # Pusty tytuł, będziemy go dodawać ręcznie
            volume=True,
            addplot=apds,
            mav=(20, 50, 100),
            figsize=(14, 10),
            tight_layout=False,  # Wyłącz tight_layout, aby mieć większą kontrolę nad rozmieszczeniem
            ylim=(min_boll_low, max_boll_high),
            returnfig=True  # Zwraca fig i axes
        )
    except Exception as e:
        st.error(f"Błąd podczas rysowania wykresu dla {ticker}: {e}")
        return None

    # Dodanie tytułu ręcznie i przesunięcie go wyżej
    fig.suptitle(
        f'Analiza Techniczna dla {ticker} (Score: {score}/{max_score}) - Ostatnie 6 Miesięcy',
        fontsize=16,
        y=0.95  # Przesunięcie tytułu wyżej
    )

    # Dostosowanie marginesów
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)

    # Dodanie buforu do ylim
    buffer = (max_boll_high - min_boll_low) * 0.02  # 2% buforu
    fig.axes[0].set_ylim(min_boll_low - buffer, max_boll_high + buffer)

    # Zwrócenie figury do wyświetlenia w Streamlit
    return fig

# Funkcja do wyświetlania raportu dla danej firmy
def display_report(ticker, data, score, max_score):
    st.subheader(f"Raport dla: {ticker}")

    # Wyświetlenie wyniku punktacji
    st.markdown(f"**Punktacja:** {score}/{max_score}")

    # Wyświetlenie wykresu
    fig = plot_stock(ticker, data, score, max_score)
    if fig:
        st.pyplot(fig)

    # Separator
    st.markdown("---")

# Główna część aplikacji Streamlit
def main():
    st.title("Analiza Techniczna Cen Akcji")

    st.markdown("""
    Ta aplikacja analizuje dane techniczne wybranych akcji i prezentuje wyniki w formie raportu.
    Możesz wprowadzić do 10 tickerów (Yahoo Finance) firm, które chcesz porównać. Im więcej punktów, tym silniejszy sygnał kupna.
    """)

    # Definicja wag dla wskaźników
    indicators_weights = {
        'RSI': {
            'buy': 1,
            'sell': -1,
            'neutral': 0
        },
        'SMA': {
            'bullish': 3,
            'bearish': -3,
            'neutral': 0
        },
        'EMA': {
            'bullish': 2,
            'bearish': -2,
            'neutral': 0
        },
        'MACD': {
            'bullish': 2,
            'bearish': -2,
            'neutral': 0
        },
        'Bollinger': {
            'buy': 1,
            'sell': -1,
            'neutral': 0
        },
        'Stochastic': {
            'buy': 1,
            'sell': -1,
            'neutral': 0
        },
        'ATR': {
            'high_volatility': 1,
            'low_volatility': -1
        },
        'OBV': {
            'bullish': 1,
            'bearish': -1,
            'neutral': 0
        }
    }

    # Oblicz maksymalną możliwą punktację
    max_score = sum(
        max(actions.values()) for actions in indicators_weights.values() if max(actions.values()) > 0
    )

    st.markdown(f"**Maksymalna możliwa punktacja:** {max_score}")

    # Formularz do wprowadzania tickerów
    with st.form("ticker_form"):
        tickers_input = st.text_area(
            "Wprowadź tickery firm do analizy (oddzielone przecinkami, maksymalnie 10):",
            value="AAPL, MSFT, GOOGL, AMZN, CDR.WA",
            help="Przykład: AAPL, MSFT, GOOGL"
        )
        submit_button = st.form_submit_button(label="Analizuj")

    if submit_button:
        # Przetwarzanie wprowadzonych tickerów
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
        tickers = [ticker for ticker in tickers if ticker]  # Usunięcie pustych wpisów
        if len(tickers) > 10:
            st.error("Możesz wprowadzić maksymalnie 10 tickerów.")
            tickers = tickers[:10]

        if not tickers:
            st.error("Wprowadź co najmniej jeden ticker.")
        else:
            st.success(f"Analizowanie tickerów: {', '.join(tickers)}")

            # Analiza każdej akcji i wyświetlanie raportu
            scores = {}
            for ticker in tickers:
                data, score = analyze_stock(ticker, indicators_weights)
                if data is not None:
                    scores[ticker] = score
                    display_report(ticker, data, score, max_score)

            # Wyświetlenie porównania punktacji
            if scores:
                st.header("Porównanie Punktacji")
                score_df = pd.DataFrame(list(scores.items()), columns=['Ticker', 'Score'])
                st.bar_chart(score_df.set_index('Ticker'))
                st.markdown("Powyższy wykres przedstawia porównanie firm pod względem liczby zdobytych punktów w analizie technicznej.\
                             Im więcej punktów, tym mocniejszy sygnał do kupna akcji.")

if __name__ == "__main__":
    main()
