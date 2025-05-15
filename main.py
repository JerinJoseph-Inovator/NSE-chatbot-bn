from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import yfinance as yf
import os
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse

# Load .env
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# FastAPI App
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

def log(msg): print(f"[LOG] {msg}")

# ------- STOCK PRICE --------
def get_stock_price(symbol: str):
    try:
        symbol = symbol.upper()
        if not symbol.endswith(".NS"):
            symbol += ".NS"
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        if not data.empty:
            latest_price = data['Close'].iloc[-1]
            return f"The current market price of {symbol.replace('.NS', '')} is ₹{latest_price:.2f} on NSE."
        return f"📉 Sorry, couldn't retrieve NSE data for {symbol.replace('.NS', '')}."
    except Exception as e:
        return f"Error fetching stock data: {str(e)}"

# ------- FUNDAMENTALS --------
def get_fundamentals(symbol: str):
    try:
        symbol = symbol.upper()
        if not symbol.endswith(".NS"):
            symbol += ".NS"
        stock = yf.Ticker(symbol)
        info = stock.info
        return (
            f"📊 Fundamental data for {symbol.replace('.NS', '')}:\n"
            f"• Market Cap: ₹{info.get('marketCap', 'N/A'):,}\n"
            f"• P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
            f"• EPS: ₹{info.get('trailingEps', 'N/A')}\n"
            f"• 52 Week High: ₹{info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"• 52 Week Low: ₹{info.get('fiftyTwoWeekLow', 'N/A')}"
        )
    except Exception as e:
        return f"Error fetching fundamentals: {str(e)}"

# ------- HISTORICAL --------
def get_historical_prices(symbol: str):
    try:
        symbol = symbol.upper()
        if not symbol.endswith(".NS"):
            symbol += ".NS"
        stock = yf.Ticker(symbol)
        data = stock.history(period="1mo")
        if data.empty:
            return "📉 No historical data found."
        summary = "\n".join(
            f"{idx.strftime('%Y-%m-%d')}: ₹{price:.2f}"
            for idx, price in data['Close'].tail(5).items()
        )
        return f"📈 Last 5 closing prices for {symbol.replace('.NS', '')}:\n{summary}"
    except Exception as e:
        return f"Error fetching historical data: {str(e)}"

# ------- GAINERS / LOSERS --------
def get_top_gainers():
    try:
        # Define a curated list of Nifty 50 stock symbols
        nifty_symbols = [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "ITC.NS", "LT.NS", "SBIN.NS", "KOTAKBANK.NS",
            "WIPRO.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BHARTIARTL.NS",
            "ULTRACEMCO.NS", "TITAN.NS", "HCLTECH.NS", "NTPC.NS", "POWERGRID.NS"
        ]

        # Fetch one-day price data
        data = yf.download(
            tickers=nifty_symbols,
            period="1d",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False
        )

        gainers = []

        for symbol in nifty_symbols:
            try:
                df = data[symbol]
                open_price = df["Open"].iloc[0]
                close_price = df["Close"].iloc[0]
                percent_change = ((close_price - open_price) / open_price) * 100

                gainers.append({
                    "symbol": symbol.replace(".NS", ""),
                    "price": round(close_price, 2),
                    "change": round(percent_change, 2)
                })

            except Exception:
                continue  # Skip any symbols with missing or bad data

        # Sort by highest gainers
        gainers.sort(key=lambda x: x["change"], reverse=True)
        top_5 = gainers[:5]

        # Format the output with padding for alignment
        output_lines = ["📈 Top Gainers Today:"]
        for stock in top_5:
            line = f"• {stock['symbol']:<10} ₹{stock['price']:.2f}  ({stock['change']:+.2f}%)"
            output_lines.append(line)

        return "\n".join(output_lines)

    except Exception as e:
        return f"⚠️ Error fetching top gainers: {str(e)}"


# ------- IPO / EARNINGS --------
def get_ipo_earnings():
    today = datetime.now()
    upcoming = today + timedelta(days=15)
    return (
        f"📅 Upcoming events:\n"
        f"• LIC IPO – Opening on {today.strftime('%d %b')} (demo data)\n"
        f"• TCS Q4 Earnings – Expected by {upcoming.strftime('%d %b')}\n"
        f"🛑 Currently this feature is in devlopment stage"
    )

# ------- NEWS --------
# Get financial news
def get_news():
    if not NEWS_API_KEY:
        return "News API key is not set."

    try:
        url = "https://newsapi.org/v2/everything"

        keywords = [
            "National Stock Exchange",
            "NSE India",
            "NSE stocks",
            "Nifty 50",
            "Sensex",
            "Indian equity market",
            "NSE listed companies",
            "stock trading India",
            "SEBI",
            "stock price India"
        ]

        query = " OR ".join(keywords)

        # Trusted Indian financial news domains
        indian_finance_domains = ",".join([
            "moneycontrol.com",
            "livemint.com",
            "economictimes.indiatimes.com",
            "business-standard.com",
            "thehindubusinessline.com",
            "financialexpress.com",
            "cnbctv18.com",
            "ndtvprofit.com"
        ])

        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 5,
            "domains": indian_finance_domains
        }

        res = requests.get(url, params=params)
        res.raise_for_status()
        articles = res.json().get("articles", [])

        if not articles:
            return "📉 No recent Indian stock market news found."

        return "\n\n".join([
            f"📰 {a['title']}\n🔗 {a['url']}"
            for a in articles
            if "NSE" in a['title'] or "Nifty" in a['title'] or "Sensex" in a['title']
        ]) or "📉 No recent relevant NSE news articles found."

    except Exception as e:
        return f"⚠️ Error fetching news: {str(e)}"

# ------- SYMBOL EXTRACTOR --------
def extract_stock_symbol(message: str):
    match = re.search(r'(?:price of|stock price of|chart of|fundamentals of)\s+([A-Za-z0-9.-]+)', message.lower())
    return match.group(1).upper() if match else None

# ------- MAIN CHAT ROUTE --------

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        user_input = body.get("message", "").strip()
        log(f"User: {user_input}")

        symbol = extract_stock_symbol(user_input)

        # Routing
        if "fundamentals" in user_input:
            return JSONResponse(content={"reply": get_fundamentals(symbol)})
        if "chart" in user_input or "historical" in user_input:
            return JSONResponse(content={"reply": get_historical_prices(symbol)})
        if "gainers" in user_input or "losers" in user_input:
            return JSONResponse(content={"reply": get_top_gainers()})
        if "ipo" in user_input or "earnings" in user_input:
            return JSONResponse(content={"reply": get_ipo_earnings()})
        if "news" in user_input:
            return JSONResponse(content={"reply": get_news()})
        if symbol:
            return JSONResponse(content={"reply": get_stock_price(symbol)})

        # Fallback to LLM
        prompt = (
            "You are an intelligent assistant from the National Stock Exchange of India (NSE). "
            "Answer factually and concisely about Indian stocks, indices, regulations, and financial data.\n"
            f"User: {user_input}\nAssistant:"
        )
        res = requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "prompt": prompt, "stream": False})
        res.raise_for_status()
        data = res.json()
        return JSONResponse(content={"reply": data.get("response", "[LLM no response]")})

    except Exception as e:
        return JSONResponse(content={"reply": f"⚠️ NSE backend error: {str(e)}"})
