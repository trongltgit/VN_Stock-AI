"""
VN Stock AI - Multi-Agent Financial Analysis System
Backend: Flask + Groq (DeepSeek-R1) + Gemini + DuckDuckGo
Hoàn toàn miễn phí
"""

import os
import uuid
import json
import tempfile
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_DIR = Path(tempfile.gettempdir()) / "vnstock_ai"
TEMP_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# AGENT 1: NEWS COLLECTOR (DuckDuckGo - Free)
# ─────────────────────────────────────────────
class NewsAgent:
    def __init__(self):
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            self.available = True
        except Exception as e:
            logger.warning(f"DuckDuckGo not available: {e}")
            self.available = False

    def search_market_news(self, query: str, max_results: int = 8) -> list:
        if not self.available:
            return []
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    f"{query} chứng khoán tài chính Việt Nam 2024 2025",
                    region="vn-vi",
                    max_results=max_results
                ))
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    def search_forex_news(self, currency_pair: str, max_results: int = 8) -> list:
        if not self.available:
            return []
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    f"tỷ giá {currency_pair} phân tích kỹ thuật 2025",
                    region="vn-vi",
                    max_results=max_results
                ))
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo forex search error: {e}")
            return []


# ─────────────────────────────────────────────
# AGENT 2: DOCUMENT READER (Gemini - Free)
# ─────────────────────────────────────────────
class DocumentAgent:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY_STOCK", "")
        self.available = bool(api_key)
        if self.available:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("Gemini DocumentAgent initialized")
            except Exception as e:
                logger.error(f"Gemini init error: {e}")
                self.available = False

    def extract_pdf_text(self, pdf_path: str) -> str:
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text[:50000]  # giới hạn token
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ""

    def analyze_document(self, text: str, symbol: str) -> dict:
        if not self.available or not text:
            return {"summary": "Không có dữ liệu tài liệu", "key_metrics": {}}
        try:
            import google.generativeai as genai
            prompt = f"""Bạn là chuyên gia phân tích tài chính. Phân tích tài liệu sau về mã {symbol}.
Trích xuất và tóm tắt:
1. Doanh thu, lợi nhuận, EPS, P/E, ROE, ROA, Debt/Equity
2. Tăng trưởng so với kỳ trước
3. Điểm mạnh và rủi ro chính
4. Nhận xét sức khỏe tài chính

Tài liệu:
{text[:20000]}

Trả lời bằng tiếng Việt, chuyên nghiệp, súc tích."""
            response = self.model.generate_content(prompt)
            return {
                "summary": response.text,
                "key_metrics": self._extract_metrics(response.text)
            }
        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return {"summary": f"Lỗi phân tích: {str(e)}", "key_metrics": {}}

    def _extract_metrics(self, text: str) -> dict:
        metrics = {}
        import re
        patterns = {
            "PE": r"P/E[:\s]*([0-9.]+)",
            "ROE": r"ROE[:\s]*([0-9.]+)%?",
            "ROA": r"ROA[:\s]*([0-9.]+)%?",
            "EPS": r"EPS[:\s]*([0-9,]+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metrics[key] = match.group(1)
        return metrics


# ─────────────────────────────────────────────
# AGENT 3: REASONING & VALUATION (Groq/DeepSeek - Free)
# ─────────────────────────────────────────────
class ReasoningAgent:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY_STOCK", "")
        self.available = bool(api_key)
        self.client = None
        if self.available:
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
                logger.info("Groq ReasoningAgent initialized")
            except Exception as e:
                logger.error(f"Groq init error: {e}")
                self.available = False

    def _call_groq(self, system_prompt: str, user_prompt: str, model: str = "deepseek-r1-distill-llama-70b") -> str:
        if not self.available:
            return "Groq API chưa được cấu hình. Vui lòng thêm GROQ_API_KEY."
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            return completion.choices[0].message.content
        except Exception as e:
            # fallback to llama
            try:
                completion = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=4096,
                )
                return completion.choices[0].message.content
            except Exception as e2:
                logger.error(f"Groq error: {e2}")
                return f"Lỗi AI: {str(e2)}"

    def analyze_stock(self, symbol: str, news_data: list, doc_analysis: dict, stock_type: str = "stock") -> dict:
        system_prompt = """Bạn là chuyên gia phân tích chứng khoán hàng đầu Việt Nam với 20 năm kinh nghiệm.
Nhiệm vụ: Phân tích toàn diện, chuyên sâu và đưa ra khuyến nghị đầu tư cụ thể.
Phong cách: Chuyên nghiệp, có căn cứ dữ liệu, rõ ràng, có tư duy phản biện."""

        news_summary = "\n".join([f"- {n.get('title','')}: {n.get('body','')[:200]}" for n in news_data[:5]])
        doc_summary = doc_analysis.get("summary", "Không có dữ liệu tài liệu")
        label = "chứng chỉ quỹ" if stock_type == "fund" else "cổ phiếu"

        user_prompt = f"""Phân tích chuyên sâu {label} {symbol} theo cấu trúc sau:

## DỮ LIỆU ĐẦU VÀO
**Tin tức thị trường:**
{news_summary or 'Không có tin tức'}

**Phân tích tài liệu tài chính:**
{doc_summary}

---
## YÊU CẦU BÁO CÁO PHÂN TÍCH

### 1. TỔNG QUAN THỊ TRƯỜNG
- Bối cảnh kinh tế vĩ mô tác động đến {symbol}
- Tâm lý thị trường hiện tại
- Ngành/lĩnh vực liên quan

### 2. PHÂN TÍCH CƠ BẢN
- Sức khỏe tài chính (thanh khoản, nợ, dòng tiền)
- Hiệu quả hoạt động kinh doanh
- Định giá (P/E, P/B, EV/EBITDA so sánh ngành)
- Lợi thế cạnh tranh và rủi ro

### 3. PHÂN TÍCH KỸ THUẬT
- Xu hướng giá ngắn/trung/dài hạn
- Các mức hỗ trợ và kháng cự quan trọng
- Tín hiệu từ các chỉ báo (RSI, MACD, MA)
- Mô hình nến và pattern đáng chú ý

### 4. PHÂN TÍCH GIÁ & ĐỊNH GIÁ
- Giá hợp lý (Fair Value) ước tính
- Vùng giá mục tiêu 3-6 tháng
- Rủi ro downside

### 5. KHUYẾN NGHỊ ĐẦU TƯ
**[MUA / BÁN / GIỮ / THEO DÕI]**
- Lý do cụ thể
- Điều kiện vào lệnh
- Stop-loss đề xuất
- Tỷ lệ phân bổ danh mục gợi ý (%)
- Thời hạn nắm giữ

### 6. RỦI RO CẦN THEO DÕI
- Top 3 rủi ro chính
- Các sự kiện catalyst sắp tới

Trả lời đầy đủ, chuyên nghiệp bằng tiếng Việt. Dùng emoji phù hợp cho dễ đọc."""

        analysis = self._call_groq(system_prompt, user_prompt)
        recommendation = self._extract_recommendation(analysis)

        return {
            "analysis": analysis,
            "recommendation": recommendation,
            "symbol": symbol,
            "type": stock_type
        }

    def analyze_forex(self, pair: str, news_data: list) -> dict:
        system_prompt = """Bạn là chuyên gia phân tích ngoại hối (Forex) với chuyên môn sâu về thị trường Việt Nam và quốc tế.
Phân tích toàn diện, chuyên nghiệp, dựa trên dữ liệu thực tế."""

        news_summary = "\n".join([f"- {n.get('title','')}: {n.get('body','')[:200]}" for n in news_data[:5]])
        base, quote = pair.replace(".", "/").split("/") if "." in pair or "/" in pair else (pair[:3], pair[3:])

        user_prompt = f"""Phân tích chuyên sâu cặp tỷ giá {base}/{quote} theo cấu trúc:

## TIN TỨC LIÊN QUAN
{news_summary or 'Không có dữ liệu tin tức'}

---
### 1. TỔNG QUAN CẶP TIỀN {base}/{quote}
- Đặc điểm và tầm quan trọng của cặp tiền này
- Bối cảnh kinh tế hai nước/khu vực

### 2. PHÂN TÍCH CƠ BẢN
- Chính sách tiền tệ NHTW liên quan
- Lạm phát, lãi suất, GDP ảnh hưởng
- Cán cân thương mại và dòng vốn
- Yếu tố địa chính trị

### 3. PHÂN TÍCH KỸ THUẬT
- Xu hướng hiện tại (ngắn/trung/dài hạn)
- Mức hỗ trợ và kháng cự chính
- RSI, MACD, Bollinger Bands
- Pattern và tín hiệu kỹ thuật

### 4. DỰ BÁO BIẾN ĐỘNG TỶ GIÁ
- Dự báo ngắn hạn (1-4 tuần)
- Dự báo trung hạn (1-3 tháng)
- Vùng mục tiêu tăng/giảm

### 5. TƯ VẤN
**[TỶ GIÁ DỰ KIẾN TĂNG / GIẢM / ĐI NGANG]**
- Chiến lược cho doanh nghiệp xuất/nhập khẩu
- Chiến lược cho nhà đầu tư cá nhân
- Mức chốt lời / cắt lỗ gợi ý

### 6. RỦI RO & CÁC SỰ KIỆN CẦN THEO DÕI

Trả lời chuyên nghiệp bằng tiếng Việt với emoji phù hợp."""

        analysis = self._call_groq(system_prompt, user_prompt)
        direction = self._extract_forex_direction(analysis)

        return {
            "analysis": analysis,
            "direction": direction,
            "pair": pair
        }

    def _extract_recommendation(self, text: str) -> str:
        text_upper = text.upper()
        if "MUA" in text_upper and ("KHUYẾN NGHỊ" in text_upper or "**MUA**" in text_upper):
            return "BUY"
        elif "BÁN" in text_upper and ("KHUYẾN NGHỊ" in text_upper or "**BÁN**" in text_upper):
            return "SELL"
        elif "GIỮ" in text_upper:
            return "HOLD"
        return "WATCH"

    def _extract_forex_direction(self, text: str) -> str:
        text_upper = text.upper()
        if "TĂNG" in text_upper:
            return "UP"
        elif "GIẢM" in text_upper:
            return "DOWN"
        return "SIDEWAYS"


# ─────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────
class AnalysisOrchestrator:
    def __init__(self):
        self.news_agent = NewsAgent()
        self.doc_agent = DocumentAgent()
        self.reasoning_agent = ReasoningAgent()

    def run_stock_analysis(self, symbol: str, pdf_paths: list = None, stock_type: str = "stock") -> dict:
        logger.info(f"Starting analysis for {symbol}")

        # Agent 1: News
        news_data = self.news_agent.search_market_news(symbol)

        # Agent 2: Documents
        doc_analysis = {"summary": "", "key_metrics": {}}
        if pdf_paths:
            combined_text = ""
            for path in pdf_paths:
                text = self.doc_agent.extract_pdf_text(path)
                combined_text += text + "\n\n"
                try:
                    os.remove(path)
                except:
                    pass
            if combined_text.strip():
                doc_analysis = self.doc_agent.analyze_document(combined_text, symbol)

        # Agent 3: Reasoning
        result = self.reasoning_agent.analyze_stock(symbol, news_data, doc_analysis, stock_type)
        result["news_count"] = len(news_data)
        result["has_documents"] = bool(pdf_paths)
        return result

    def run_forex_analysis(self, pair: str) -> dict:
        logger.info(f"Starting forex analysis for {pair}")
        news_data = self.news_agent.search_forex_news(pair)
        result = self.reasoning_agent.analyze_forex(pair, news_data)
        result["news_count"] = len(news_data)
        return result


orchestrator = AnalysisOrchestrator()


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "agents": {
            "news": orchestrator.news_agent.available,
            "document": orchestrator.doc_agent.available,
            "reasoning": orchestrator.reasoning_agent.available
        }
    })

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        symbol = request.form.get("symbol", "").strip().upper()
        analysis_type = request.form.get("type", "stock")  # stock | fund | forex
        urls = request.form.get("urls", "")

        if not symbol:
            return jsonify({"error": "Vui lòng nhập mã chứng khoán hoặc cặp tiền tệ"}), 400

        # Forex analysis
        if analysis_type == "forex" or ("." in symbol and len(symbol) <= 7):
            result = orchestrator.run_forex_analysis(symbol)
            return jsonify({"success": True, "data": result, "mode": "forex"})

        # Save uploaded PDFs
        pdf_paths = []
        if "pdfs" in request.files:
            files = request.files.getlist("pdfs")
            for f in files:
                if f and f.filename.endswith(".pdf"):
                    tmp_path = str(TEMP_DIR / f"{uuid.uuid4()}.pdf")
                    f.save(tmp_path)
                    pdf_paths.append(tmp_path)

        result = orchestrator.run_stock_analysis(symbol, pdf_paths, analysis_type)
        return jsonify({"success": True, "data": result, "mode": "stock"})

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/scrape-url", methods=["POST"])
def scrape_url():
    """Scrape financial data from VCBS or VCBF URLs"""
    try:
        data = request.get_json()
        url = data.get("url", "")
        if not url:
            return jsonify({"error": "URL không hợp lệ"}), 400

        import requests as req
        from bs4 import BeautifulSoup
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = req.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")

        # Extract text content
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)[:5000]

        return jsonify({"success": True, "content": text, "url": url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
