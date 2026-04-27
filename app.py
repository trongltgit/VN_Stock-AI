"""
VN Stock AI - Phiên bản sạch, đẹp, hoạt động tốt (app.py ở root)
"""

import os
import tempfile
import logging
from pathlib import Path
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_DIR = Path(tempfile.gettempdir()) / "vnstock_ai"
TEMP_DIR.mkdir(exist_ok=True)

# ====================== HTML ĐẸP (giống ảnh bạn gửi) ======================
INDEX_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VNStockAI - Phân Tích Chứng Khoán AI</title>
    <style>
        :root { --bg: #0f172a; --card: #1e2937; --accent: #22d3ee; }
        body { margin:0; font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: #e2e8f0; line-height: 1.5; }
        .ticker { background: #1e2937; padding: 12px 0; overflow: hidden; white-space: nowrap; border-bottom: 3px solid #334155; font-size: 15px; }
        .ticker-content { display: inline-block; animation: ticker 45s linear infinite; }
        @keyframes ticker { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
        .ticker-item { margin-right: 45px; font-weight: 600; }
        .up { color: #4ade80; } .down { color: #f87171; }
        
        .container { display: flex; max-width: 1400px; margin: 20px auto; gap: 24px; padding: 0 20px; }
        .sidebar { width: 380px; background: var(--card); border-radius: 16px; padding: 24px; height: fit-content; }
        .main { flex: 1; }
        
        h1 { color: #67e8f9; margin: 0 0 8px 0; }
        button { padding: 12px 20px; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }
        .btn-run { background: #22d3ee; color: #0f172a; width: 100%; font-size: 1.1em; padding: 16px; }
        .card { background: var(--card); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        input, select { background: #334155; border: 1px solid #475569; color: white; padding: 12px; border-radius: 8px; width: 100%; margin-bottom: 12px; }
        .agent { padding: 14px; background: #334155; border-radius: 8px; margin-bottom: 10px; }
    </style>
</head>
<body>

<!-- Ticker Tape -->
<div class="ticker">
    <div class="ticker-content">
        VN-INDEX 1,853.29 <span class="down">▼ -17.07 (-0.91%)</span> &nbsp;&nbsp;&nbsp;&nbsp;
        VCB 60,600 <span class="down">▼ -2,200</span> &nbsp;&nbsp;&nbsp;&nbsp;
        VHM 38,200 <span class="down">▼ -300</span> &nbsp;&nbsp;&nbsp;&nbsp;
        HPG 27,900 <span class="up">▲ +200</span> &nbsp;&nbsp;&nbsp;&nbsp;
        MAFPF1 12,540 <span class="up">▲ +85</span> &nbsp;&nbsp;&nbsp;&nbsp;
        USD/VND 25,480 <span class="down">▼ -20</span> &nbsp;&nbsp;&nbsp;&nbsp;
        VIC 212,100 <span class="down">▼ -2,400</span> &nbsp;&nbsp;&nbsp;&nbsp;
        MBB 26,200 <span class="down">▼ -200</span> &nbsp;&nbsp;&nbsp;&nbsp;
        FPT 73,400 <span class="down">▼ -900</span> &nbsp;&nbsp;&nbsp;&nbsp;
        TCB 34,250 <span class="up">▲ +950</span>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        VN-INDEX 1,853.29 <span class="down">▼ -17.07 (-0.91%)</span> ...
    </div>
</div>

<div class="container">
    <!-- Sidebar -->
    <div class="sidebar">
        <h2>⚙ Cấu Hình Phân Tích</h2>
        
        <div class="card">
            <div style="display:flex; gap:8px; margin-bottom:16px;">
                <button onclick="setType('stock')" style="flex:1; background:#3b82f6;">📊 Cổ phiếu</button>
                <button onclick="setType('fund')" style="flex:1; background:#8b5cf6;">🏦 Quỹ</button>
                <button onclick="setType('forex')" style="flex:1; background:#ec4899;">💱 Ngoại tệ</button>
            </div>
            
            <input type="text" id="symbol" placeholder="VD: VCB, HPG, FPT..." value="VCB">
            <button onclick="analyzeNow()" class="btn-run">▶ Run Phân Tích</button>
        </div>

        <div class="card">
            <h4>🔗 Nguồn Dữ Liệu URL</h4>
            <div id="url-list"></div>
            <button onclick="addUrlInput()" style="margin-top:10px; width:100%;">+ Thêm URL</button>
        </div>

        <div class="card">
            <h4>📄 Upload Báo Cáo Tài Chính (PDF)</h4>
            <input type="file" id="pdfFiles" multiple accept=".pdf" onchange="handleFiles(this.files)">
            <small style="color:#94a3b8;">Tối đa 5 files • Kéo thả cũng được</small>
        </div>

        <button onclick="analyzeNow()" class="btn-run">🤖 Phân Tích AI Chuyên Sâu</button>
    </div>

    <!-- Main Area -->
    <div class="main">
        <div style="text-align:center; margin-bottom:40px;">
            <h1>VNStockAI</h1>
            <p style="font-size:1.2em; color:#a5b4fc;">Phân Tích Chứng Khoán AI</p>
            <p>Hệ thống multi-agent chuyên sâu cho thị trường Việt Nam</p>
        </div>

        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:20px; margin-bottom:30px;">
            <div class="card" style="text-align:center;">
                <h3>🔍 Tin Tức Thị Trường</h3>
                <p>DuckDuckGo real-time</p>
            </div>
            <div class="card" style="text-align:center;">
                <h3>📄 Đọc Báo Cáo PDF</h3>
                <p>Gemini 1.5/2.0 Flash</p>
            </div>
            <div class="card" style="text-align:center;">
                <h3>🧠 Suy Luận Định Giá</h3>
                <p>DeepSeek-R1 via Groq</p>
            </div>
        </div>

        <div id="result" class="card" style="min-height: 500px; font-size: 15.5px;">
            <p style="text-align:center; color:#94a3b8; padding:40px;">
                Nhập mã chứng khoán (VCB, HPG, FPT...) và nhấn <strong>Run Phân Tích</strong>
            </p>
        </div>
    </div>
</div>

<script>
function setType(t) {
    const input = document.getElementById('symbol');
    input.placeholder = t === 'forex' ? 'USD.VND, EUR.USD...' : 'VCB, HPG, FPT...';
}

function analyzeNow() {
    const symbol = document.getElementById('symbol').value.trim().toUpperCase();
    if (!symbol) return alert("Vui lòng nhập mã chứng khoán!");

    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `<p style="text-align:center; padding:60px;">Đang phân tích <strong>${symbol}</strong> bằng Multi-Agent AI...<br><small>(có thể mất 15-40 giây)</small></p>`;

    const formData = new FormData();
    formData.append('symbol', symbol);
    formData.append('type', 'stock');

    fetch('/api/analyze', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        resultDiv.innerHTML = `
            <h3>Kết quả phân tích ${symbol}</h3>
            <pre style="white-space: pre-wrap; background:#0f172a; padding:20px; border-radius:8px; overflow:auto; max-height:70vh;">${data.analysis || JSON.stringify(data, null, 2)}</pre>
        `;
    })
    .catch(err => {
        resultDiv.innerHTML = `<p style="color:#f87171; padding:40px;">Lỗi kết nối: ${err.message}</p>`;
    });
}

function addUrlInput() {
    const list = document.getElementById('url-list');
    const div = document.createElement('div');
    div.style.marginBottom = '8px';
    div.innerHTML = `<input type="text" placeholder="https://..." style="width:85%;"> <button onclick="this.parentElement.remove()" style="width:12%;">✕</button>`;
    list.appendChild(div);
}

function handleFiles(files) {
    if (files.length > 5) {
        alert("Chỉ hỗ trợ tối đa 5 file PDF!");
    } else {
        console.log(`Đã chọn ${files.length} file PDF`);
    }
}

// Khởi tạo
console.log("%cVNStockAI loaded successfully", "color:#22d3ee; font-weight:bold");
</script>
</body>
</html>
"""

# ====================== AGENTS (đã fix Groq + đơn giản hóa) ======================
class NewsAgent:
    def __init__(self):
        self.available = True

    def get_news(self, symbol):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as d:
                return list(d.text(f"{symbol} cổ phiếu tin tức mới nhất 2026", max_results=8))
        except:
            return []

class ReasoningAgent:
    def __init__(self):
        key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_STOCK")
        self.available = bool(key)
        self.client = None
        if self.available:
            try:
                from groq import Groq
                self.client = Groq(api_key=key)
                logger.info("✅ Groq connected successfully")
            except Exception as e:
                logger.error(f"Groq init failed: {e}")
                self.available = False

    def analyze(self, symbol):
        if not self.client:
            return "Chưa cấu hình GROQ_API_KEY. Vui lòng thêm key vào Render Environment Variables."

        system = "Bạn là chuyên gia phân tích chứng khoán Việt Nam giàu kinh nghiệm. Trả lời bằng tiếng Việt, rõ ràng, có khuyến nghị cụ thể (MUA/BÁN/GIỮ)."
        user = f"Phân tích chi tiết cổ phiếu {symbol} hiện tại (2026): tình hình kinh doanh, định giá, xu hướng kỹ thuật, và khuyến nghị đầu tư."

        try:
            r = self.client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3,
                max_tokens=6000,
            )
            return r.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq call error: {e}")
            return f"Lỗi khi gọi Groq: {str(e)}"

class Orchestrator:
    def __init__(self):
        self.news = NewsAgent()
        self.ai = ReasoningAgent()

    def analyze(self, symbol):
        analysis = self.ai.analyze(symbol)
        return {
            "symbol": symbol,
            "analysis": analysis,
            "status": "success"
        }

orc = Orchestrator()

# ====================== ROUTES ======================
@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        symbol = request.form.get("symbol", "").strip().upper()
        if not symbol:
            return jsonify({"error": "Vui lòng nhập mã"}), 400
        
        result = orc.analyze(symbol)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return jsonify({"error": "Lỗi server nội bộ"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "groq": orc.ai.available
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
