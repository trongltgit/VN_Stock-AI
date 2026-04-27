"""
VN Stock AI - Fixed version
- Wrap response in {mode, data} to match frontend (json.data.recommendation)
- Accept but ignore urls/pdfs from frontend FormData
"""

import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================== AGENTS ======================
class NewsAgent:
    def __init__(self):
        self.available = True

    def get_news(self, symbol):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as d:
                return list(d.text(f"{symbol} cổ phiếu tin tức 2026", max_results=8))
        except:
            return []

class ReasoningAgent:
    MODELS = ["llama-3.3-70b-versatile", "llama3-70b-8192"]

    def __init__(self):
        key = os.getenv("GROQ_API_KEY_STOCK")
        self.available = bool(key)
        self.client = None
        if self.available:
            try:
                from groq import Groq
                self.client = Groq(api_key=key)
                logger.info("✅ Groq initialized successfully with GROQ_API_KEY_STOCK")
            except Exception as e:
                logger.error(f"Groq init error: {e}")
                self.available = False

    def analyze(self, symbol, stock_type="stock"):
        if not self.client:
            return {"analysis": "❌ Chưa cấu hình GROQ_API_KEY_STOCK trên Render.com", "recommendation": "WATCH"}

        system = """Bạn là chuyên gia phân tích chứng khoán Việt Nam. 
Trả lời bằng tiếng Việt, chuyên nghiệp, có cấu trúc rõ ràng.
Luôn đưa ra khuyến nghị: **[MUA / BÁN / GIỮ]** + giá mục tiêu + luận điểm."""

        user = f"""Phân tích chi tiết {stock_type} **{symbol}** năm 2026:
- Tóm tắt tình hình kinh doanh
- Định giá (P/E, P/B, fair value)
- Phân tích kỹ thuật ngắn hạn/trung hạn
- Khuyến nghị đầu tư cụ thể (MUA/BÁN/GIỮ) kèm giá mục tiêu 3-6 tháng
- Rủi ro chính"""

        for model in self.MODELS:
            try:
                r = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0.3,
                    max_tokens=5500,
                )
                analysis = r.choices[0].message.content

                rec = "HOLD"
                upper = analysis.upper()
                if any(x in upper for x in ["MUA", "BUY"]): rec = "BUY"
                elif any(x in upper for x in ["BÁN", "SELL"]): rec = "SELL"

                return {"analysis": analysis, "recommendation": rec}
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                continue

        return {"analysis": "Lỗi kết nối Groq. Vui lòng thử lại sau.", "recommendation": "WATCH"}

class Orchestrator:
    def __init__(self):
        self.news = NewsAgent()
        self.ai = ReasoningAgent()

    def analyze(self, symbol, stock_type="stock"):
        result = self.ai.analyze(symbol, stock_type)
        # ✅ FIX: Wrap in {mode, data} to match frontend (const data = json.data)
        return {
            "mode": stock_type,
            "data": {
                "symbol": symbol,
                "type": stock_type,
                "analysis": result["analysis"],
                "recommendation": result["recommendation"],
                "news_count": 0,
                "has_documents": False
            }
        }

orc = Orchestrator()

# ====================== ROUTES ======================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        symbol = request.form.get("symbol", "").strip()
        stock_type = request.form.get("type", "stock")
        # ✅ FIX: Frontend also sends urls/pdfs — accept but ignore for now
        urls = request.form.get("urls", "[]")
        logger.info(f"Analyze: symbol={symbol}, type={stock_type}")

        if not symbol:
            return jsonify({"error": "Vui lòng nhập mã chứng khoán"}), 400

        symbol = symbol.upper()
        result = orc.analyze(symbol, stock_type)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Analyze error: {e}", exc_info=True)
        return jsonify({"error": f"Lỗi server: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "groq": orc.ai.available
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
