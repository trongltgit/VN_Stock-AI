"""
VN Stock AI - Sử dụng templates/index.html + Backend phân tích chuyên sâu
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

# ====================== AGENTS - PHÂN TÍCH CHUYÊN SÂU ======================
class NewsAgent:
    def __init__(self):
        self.available = True

    def get_news(self, symbol):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as d:
                return list(d.text(f"{symbol} cổ phiếu tin tức mới nhất 2026", max_results=10))
        except Exception as e:
            logger.warning(f"NewsAgent error: {e}")
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
                logger.error(f"Groq init failed: {e}")
                self.available = False

    def analyze(self, symbol, stock_type="stock"):
        if not self.client:
            return "❌ Chưa cấu hình GROQ_API_KEY_STOCK trên Render.com"

        system = """Bạn là chuyên gia phân tích chứng khoán cấp cao tại Việt Nam (CFA). 
Trả lời bằng tiếng Việt, phong cách chuyên nghiệp như báo cáo của SSI, VCSC, Maybank. 
Luôn đưa ra khuyến nghị rõ ràng: MUA / BÁN / GIỮ kèm giá mục tiêu, rủi ro và luận điểm cụ thể."""

        user = f"""Phân tích chuyên sâu { 'cổ phiếu' if stock_type == 'stock' else 'chứng chỉ quỹ' if stock_type == 'fund' else 'tỷ giá ngoại tệ' } **{symbol}** năm 2026.
Bao gồm: 
- Tình hình tài chính & tăng trưởng
- Định giá (P/E, P/B, fair value)
- Phân tích kỹ thuật (xu hướng, hỗ trợ/kháng cự, RSI, MACD)
- Khuyến nghị đầu tư cụ thể (MUA/BÁN/GIỮ) + giá mục tiêu 3-6 tháng
- Rủi ro chính và catalyst"""

        try:
            r = self.client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.25,
                max_tokens=6000,
            )
            return r.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq analyze error: {e}")
            return f"Lỗi khi gọi DeepSeek-R1: {str(e)}"

class Orchestrator:
    def __init__(self):
        self.news = NewsAgent()
        self.ai = ReasoningAgent()

    def analyze(self, symbol, stock_type="stock"):
        news = self.news.get_news(symbol)
        analysis = self.ai.analyze(symbol, stock_type)
        return {
            "symbol": symbol,
            "type": stock_type,
            "analysis": analysis,
            "news_count": len(news),
            "has_documents": False   # sẽ mở rộng sau khi xử lý PDF
        }

orc = Orchestrator()

# ====================== ROUTES ======================
@app.route("/")
def index():
    return render_template("index.html")   # Sử dụng file /templates/index.html

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        symbol = request.form.get("symbol", "").strip().toUpperCase()
        stock_type = request.form.get("type", "stock")

        if not symbol:
            return jsonify({"error": "Vui lòng nhập mã"}), 400

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
