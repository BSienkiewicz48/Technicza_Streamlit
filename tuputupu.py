import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import streamlit as st

# Funkcja do analizy akcji
def analyze_stock(ticker, indicators_weights):
    # Pobierz dane historyczne z dłuższego okresu (2 lata)
    data = yf.download(ticker, period='2y', interval='1d')
    if data.empty:
        st.warning(f"Brak danych dla {ticker}")
        return None, 0
    
    # Oblicz wskaźniki techniczne
    data['RSI'] = ta.rsi(data['Close'], length=14)
    data['SMA50'] = ta.sma(data['Close'], length=50)
    data['SMA200'] = ta.sma(data['Close'], length=200)
    data['EMA20'] = ta.ema(data['Close'], length=20)
    data['EMA100'] = ta.ema(data['Close'], length=100)
    
    # MACD
    macd = ta.macd(data['Close'])
    data['MACD'] = macd['MACD_12_26_9']
    data['Signal'] = macd['MACDs_12_26_9']
    
    # Bollinger Bands
    bb = ta.bbands(data['Close'], length=20, std=2)
    data['Bollinger_Low'] = bb['BBL_20_2.0']
    data['Bollinger_Middle'] = bb['BBM_20_2.0']
    data['Bollinger_High'] = bb['BBU_20_2.0']
    
    # Stochastic Oscillator
    stoch = ta.stoch(data['High'], data['Low'], data['Close'], k=14, d=3)
    data['Stochastic_K'] = stoch['STOCHk_14_3_3']
    data['Stochastic_D'] = stoch['STOCHd_14_3_3']
    
    # ATR
    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
    
    # On-Balance Volume
    data['OBV'] = ta.obv(data['Close'], data['Volume'])
    
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
    if not np.isnan(latest_sma50) and not np.isnan(latest_sma200):
        if latest_close > latest_sma50 > latest_sma200:
            score += indicators_weights['SMA']['bullish']
        elif latest_close < latest_sma50 < latest_sma200:
            score += indicators_weights['SMA']['bearish']
        else:
            score += indicators_weights['SMA']['neutral']
    else:
        score += indicators_weights['SMA']['neutral']
    
    # EMA Analysis (Ważny)
    if not np.isnan(latest_ema20) and not np.isnan(latest_ema100):
        if latest_ema20 > latest_ema100:
            score += indicators_weights['EMA']['bullish']
        elif latest_ema20 < latest_ema100:
            score += indicators_weights['EMA']['bearish']
        else:
            score += indicators_weights['EMA']['neutral']
    else:
        score += indicators_weights['EMA']['neutral']
    
    # MACD Analysis (Ważny)
    if not np.isnan(latest_macd) and not np.isnan(latest_signal):
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
    st.title("Aplikacja Analizy Technicznej Akcji")
    
    st.markdown("""
    Ta aplikacja analizuje dane techniczne wybranych akcji i prezentuje wyniki w formie raportu.
    Możesz wprowadzić do 10 tickerów firm, które chcesz porównać.
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
            value="AAPL, MSFT, GOOGL, AMZN, TCDR.WA",
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
                
                # Opcjonalnie, można dodać tabelę z wynikami, ale użytkownik prosił o wykresy bez tabel
                # st.table(score_df)
    
if __name__ == "__main__":
    main()
