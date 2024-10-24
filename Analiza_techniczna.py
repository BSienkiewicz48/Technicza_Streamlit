import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import streamlit as st
import math

# Funkcja do obliczania RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Funkcja do obliczania MACD
def calculate_macd(series, span_short=12, span_long=26, span_signal=9):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    return macd, signal

# Funkcja do obliczania Bollinger Bands
def calculate_bollinger_bands(series, window=20, std_dev=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return sma, upper, lower

# Funkcja do obliczania Stochastic Oscillator
def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    stochastic_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    stochastic_d = stochastic_k.rolling(window=d_window).mean()
    return stochastic_k, stochastic_d

# Funkcja do obliczania ATR
def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close_prev = (high - close.shift()).abs()
    low_close_prev = (low - close.shift()).abs()
    tr = high_low.combine(high_close_prev, max).combine(low_close_prev, max)
    atr = tr.rolling(window=period).mean()
    return atr

# Funkcja do obliczania OBV
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
    
    # Oblicz wskaźniki techniczne
    data['RSI'] = calculate_rsi(data['Close'], period=14)
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
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
    latest_close = data['Close'].iloc[-1]
    latest_rsi = data['RSI'].iloc[-1]
    latest_sma50 = data['SMA50'].iloc[-1]
    latest_sma200 = data['SMA200'].iloc[-1]
    latest_ema20 = data['EMA20'].iloc[-1]
    latest_ema100 = data['EMA100'].iloc[-1]
    latest_macd = data['MACD'].iloc[-1]
    latest_signal = data['Signal'].iloc[-1]
    latest_boll_high = data['Bollinger_High'].iloc[-1]
    latest_boll_low = data['Bollinger_Low'].iloc[-1]
    latest_stoch_k = data['Stochastic_K'].iloc[-1]
    latest_stoch_d = data['Stochastic_D'].iloc[-1]
    latest_atr = data['ATR'].iloc[-1]
    latest_obv = data['OBV'].iloc[-1]
    
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
    if latest_sma50 is not None and latest_sma200 is not None and not math.isnan(latest_sma50) and not math.isnan(latest_sma200):
        if latest_close > latest_sma50 > latest_sma200:
            score += indicators_weights['SMA']['bullish']
        elif latest_close < latest_sma50 < latest_sma200:
            score += indicators_weights['SMA']['bearish']
        else:
            score += indicators_weights['SMA']['neutral']
    else:
        score += indicators_weights['SMA']['neutral']

    # EMA Analysis (Ważny)
    if latest_ema20 is not None and latest_ema100 is not None and not math.isnan(latest_ema20) and not math.isnan(latest_ema100):
        if latest_ema20 > latest_ema100:
            score += indicators_weights['EMA']['bullish']
        elif latest_ema20 < latest_ema100:
            score += indicators_weights['EMA']['bearish']
        else:
            score += indicators_weights['EMA']['neutral']
    else:
        score += indicators_weights['EMA']['neutral']

    # MACD Analysis (Ważny)
    if latest_macd is not None and latest_signal is not None and not math.isnan(latest_macd) and not math.isnan(latest_signal):
        if latest_macd > latest_signal:
            score += indicators_weights['MACD']['bullish']
        elif latest_macd < latest_signal:
            score += indicators_weights['MACD']['bearish']
        else:
            score += indicators_weights['MACD']['neutral']
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
    df_plot = df.loc[df.index >= six_months_ago]
    
    # Obliczenie minimalnej i maksymalnej wartości wstęg Bollingera
    min_boll_low = df_plot['Bollinger_Low'].min()
    max_boll_high = df_plot['Bollinger_High'].max()
    
    # Definiowanie dodatkowych wskaźników do wykresu
    apds = []
    
    # EMA20 i EMA100
    apds.append(mpf.make_addplot(df_plot['EMA20'], color='blue', width=1))
    apds.append(mpf.make_addplot(df_plot['EMA100'], color='orange', width=1))
    
    # Bollinger Bands
    apds.append(mpf.make_addplot(df_plot['Bollinger_High'], color='red', linestyle='--'))
    apds.append(mpf.make_addplot(df_plot['Bollinger_Low'], color='green', linestyle='--'))
    
    # OBV Plot (panel=2)
    obv_plot = mpf.make_addplot(df_plot['OBV'], panel=2, color='purple', ylabel='OBV')
    
    # Rysowanie wykresu świecowego z dodatkowymi wskaźnikami
    fig, axes = mpf.plot(
        df_plot,
        type='candle',
        style='charles',
        title='',  # Pusty tytuł, będziemy go dodawać ręcznie
        volume=True,
        addplot=apds + [obv_plot],
        mav=(20, 50, 100),
        figsize=(14, 10),
        tight_layout=False,  # Wyłącz tight_layout, aby mieć większą kontrolę nad rozmieszczeniem
        ylim=(min_boll_low, max_boll_high),
        returnfig=True  # Zwraca fig i axes
    )
    
    # Dodanie tytułu ręcznie i przesunięcie go wyżej
    fig.suptitle(
        f'Analiza Techniczna dla {ticker} (Score: {score}/{max_score}) - Ostatnie 6 Miesięcy',
        fontsize=16,
        y=0.95  # Przesunięcie tytułu wyżej (dostosuj wartość według potrzeb)
    )
    
    # Dostosowanie marginesów, aby wykres przesunął się w lewo i wypełnił więcej przestrzeni
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
    
    # Opcjonalny: Dodanie buforu do ylim, aby upewnić się, że etykiety nie są obcięte
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
    st.pyplot(fig)
    
    # Separator
    st.markdown("---")

# Główna część aplikacji Streamlit
def main():
    st.title("Analiza Techniczna Cen Akcji")
    
    st.markdown("""
    Ta aplikacja analizuje dane techniczne wybranych akcji i prezentuje wyniki w formie raportu.
    Możesz wprowadzić do 10 tickerów (Yahoo finance) firm, które chcesz porównać. Im więcej punktów tym silniejszy sygnał kupna.
    """)
    
    # Definicja wag dla wskaźników (ważniejsze wskaźniki mają wyższe wagi)
    indicators_weights = {
        'RSI': {
            'buy': 1,    # Mniejsza waga za RSI
            'sell': -1,
            'neutral': 0
        },
        'SMA': {
            'bullish': 3,  # Najwyższa waga za SMA
            'bearish': -3,
            'neutral': 0
        },
        'EMA': {
            'bullish': 2,  # Ważny wskaźnik
            'bearish': -2,
            'neutral': 0
        },
        'MACD': {
            'bullish': 2,
            'bearish': -2,
            'neutral': 0
        },
        'Bollinger': {
            'buy': 1,    # Mniejsza waga za Bollinger Bands
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
    max_score = 0
    for indicator, actions in indicators_weights.items():
        max_action_score = max(actions.values())
        if max_action_score > 0:
            max_score += max_action_score
    
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

                # Użycie Streamlit do wyświetlenia wykresu słupkowego
                st.bar_chart(score_df.set_index('Ticker'))
                st.markdown("Powyższy wykres przedstawia porównanie firm pod względem liczby zdobytych punktów w analizie technicznej.\
                         Im więcej punktów tym mocniejszy sygnał do kupna akcji.")
                # Opcjonalnie, można dodać tabelę z wynikami, ale użytkownik prosił o wykresy bez tabel
                # st.table(score_df)

if __name__ == "__main__":
    main()
