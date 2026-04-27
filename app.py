"""
VN Stock AI - Phiên bản sửa lỗi + dùng templates/index.html
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
            return "❌ Chưa cấu hình GROQ_API_KEY_STOCK"

        system = """Bạn là chuyên gia phân tích chứng khoán Việt Nam giàu kinh nghiệm.
Trả lời bằng tiếng Việt, rõ ràng, chuyên nghiệp. Luôn đưa ra khuyến nghị MUA/BÁN/GIỮ + giá mục tiêu."""

        user = f"""Phân tích chuyên sâu {stock_type} **{symbol}**:
- Tình hình kinh doanh và tài chính
- Định giá hợp lý
- Xu hướng kỹ thuật
- Khuyến nghị đầu tư cụ thể (MUA/BÁN/GIỮ) kèm giá mục tiêu
- Rủi ro chính"""

        try:
            r = self.client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3,
                max_tokens=5500,
            )
            return r.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return f"Lỗi Groq: {str(e)}"

class Orchestrator:
    def __init__(self):
        self.news = NewsAgent()
        self.ai = ReasoningAgent()

    def analyze(self, symbol, stock_type="stock"):
        analysis = self.ai.analyze(symbol, stock_type)
        return {
            "symbol": symbol,
            "type": stock_type,
            "analysis": analysis,
            "news_count": 0,
            "has_documents": False
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

        if not symbol:
            return jsonify({"error": "Vui lòng nhập mã chứng khoán"}), 400

        # SỬA LỖI Ở ĐÂY: Dùng .upper() thay vì .toUpperCase()
        symbol = symbol.upper()

        result = orc.analyze(symbol, stock_type)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return jsonify({"error": "Lỗi server khi phân tích"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "groq": orc.ai.available,
        "gemini": bool(os.getenv("GEMINI_API_KEY_STOCK"))
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
