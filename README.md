# 🧠 NSE Chatbot – Backend

This is the production-ready **backend** of the NSE Chatbot project, powered by **FastAPI**. It serves as the intelligent engine behind the frontend, providing real-time responses, financial data, charting, and news—using a local **Ollama** model or fallback to **OpenRouter API**.

## 🚀 Features

- ⚙️ FastAPI-powered, async backend
- 🧠 Local LLM (Mistral via Ollama) or OpenRouter API fallback
- 📊 Stock fundamentals, historical charts, gainers/losers
- 📅 IPOs, earnings calendar, and filtered NSE news
- 🔄 Switch backend model source using `.env` flag
- 🌐 Hosted on Render

## 🧰 Tech Stack

- **FastAPI** – High-performance Python web framework
- **Pydantic** – Data validation and settings management
- **httpx** – Async HTTP client for external APIs
- **yfinance** – Yahoo Finance wrapper
- **NewsAPI** – For fetching latest NSE-related news
- **Ollama** – Local Mistral-7B model interface
- **OpenRouter API** – Cloud-based fallback LLM access

## 📦 Installation

```bash
git clone https://github.com/your-username/nse-chatbot-backend.git
cd nse-chatbot-backend
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file in the root directory:

```env
USE_LOCAL_OLLAMA=true  # Set to false to use OpenRouter
OLLAMA_MODEL=mistral
OPENROUTER_MODEL=mistralai/mistral-7b-instruct:free
OPENROUTER_API_KEY=your_openrouter_api_key
```

## 🧪 Run Locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Test via browser or Postman: [http://localhost:8000/docs](http://localhost:8000/docs)

## 🧱 Project Structure

```
.
├── main.py
├── routers/
│   ├── chat.py
│   ├── fundamentals.py
│   ├── news.py
│   └── nse_data.py
├── services/
│   ├── llm_handler.py
│   ├── stock_utils.py
│   └── news_utils.py
├── .env
└── requirements.txt
```

## 🌐 Deployment

Hosted on **Render** using `uvicorn` with automatic deployment from GitHub.

1. Push your backend repo to GitHub
2. Create a new web service on [Render](https://render.com)
3. Set the start command as:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 10000
   ```
4. Add environment variables from your `.env`

## 🔐 Security

- CORS configured for frontend access
- API keys stored securely via `.env`
- Safe model switching logic to prevent abuse

## 🤝 Contributions

Pull requests welcome. For significant changes, open an issue to propose changes and discuss implementation.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

**Built for performance, extensibility, and intelligent stock market interaction.**
Developed by **Jerin Joseph Alour**  
🔗 [jerin.cloud](https://jerin.cloud)
