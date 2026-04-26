require("dotenv").config();
const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const rateLimit = require("express-rate-limit");
const fetch = require("node-fetch");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

// Security & middleware
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors());
app.use(express.json({ limit: "10kb" }));
app.use(express.static(path.join(__dirname, "public")));

// Rate limiting: 30 requests per 10 minutes per IP
const limiter = rateLimit({
  windowMs: 10 * 60 * 1000,
  max: 30,
  message: { error: "Quá nhiều yêu cầu. Vui lòng thử lại sau 10 phút." },
  standardHeaders: true,
  legacyHeaders: false,
});
app.use("/api/", limiter);

// ─── API: Analyze stock/fund ───────────────────────────────────────────────
app.post("/api/analyze", async (req, res) => {
  const { ticker, assetType, analysisTypes, sources } = req.body;

  if (!ticker || typeof ticker !== "string") {
    return res.status(400).json({ error: "Mã chứng khoán không hợp lệ." });
  }

  const cleanTicker = ticker.trim().toUpperCase().substring(0, 20);
  const typeLabel =
    assetType === "fund"
      ? "chứng chỉ quỹ"
      : assetType === "etf"
      ? "ETF"
      : assetType === "bond"
      ? "trái phiếu"
      : "cổ phiếu";

  const activeTypes = (analysisTypes || []).join(", ") || "Phân tích toàn diện";
  const activeSources = (sources || []).join(", ") || "VCBS, VCBF, CafeF, VietStock";

  const prompt = `Bạn là chuyên gia phân tích chứng khoán Việt Nam hàng đầu với 20 năm kinh nghiệm, am hiểu sâu về thị trường HOSE, HNX, UPCOM và các quỹ đầu tư Việt Nam.

Hãy thực hiện phân tích TOÀN DIỆN và CHUYÊN SÂU cho mã "${cleanTicker}" (${typeLabel}).

Các loại phân tích cần thực hiện: ${activeTypes}
Nguồn dữ liệu tham chiếu: ${activeSources}

Trả lời CHÍNH XÁC chỉ JSON (không markdown, không backtick, không giải thích thêm):
{
  "company": "Tên công ty/quỹ đầy đủ chính xác",
  "sector": "Ngành/lĩnh vực",
  "exchange": "HOSE hoặc HNX hoặc UPCOM hoặc VCBF",
  "price": "Giá gần nhất bạn biết (VND, dạng XX.XXX)",
  "price_note": "Giá tham chiếu - cần cập nhật thời gian thực",
  "change_pct": "+X.XX% hoặc -X.XX%",
  "pe_ratio": "X.X hoặc N/A",
  "pb_ratio": "X.X hoặc N/A",
  "eps": "X.XXX VND hoặc N/A",
  "roe": "XX.X% hoặc N/A",
  "market_cap": "X,XXX tỷ VND",
  "week52_high": "XX.XXX VND",
  "week52_low": "XX.XXX VND",
  "verdict": "MUA hoặc BÁN hoặc GIỮ",
  "verdict_confidence": 75,
  "target_price_3m": "XX.XXX VND",
  "target_price_1y": "XX.XXX VND",
  "stop_loss": "XX.XXX VND",
  "fundamental_score": 72,
  "technical_score": 68,
  "price_score": 65,
  "market_score": 70,
  "risk_level": "Thấp hoặc Trung bình hoặc Cao",
  "fundamental_summary": "Phân tích cơ bản 4-5 câu: doanh thu, lợi nhuận, tỷ suất sinh lời, nợ/vốn, cổ tức, triển vọng tăng trưởng...",
  "technical_summary": "Phân tích kỹ thuật 4-5 câu: MA20/50/200, RSI, MACD, Bollinger Band, khối lượng giao dịch, xu hướng...",
  "chart_summary": "Phân tích đồ thị 3-4 câu: mô hình nến, pattern, breakout/breakdown, vùng tích lũy/phân phối...",
  "price_summary": "Phân tích giá 4-5 câu: vùng hỗ trợ mạnh, kháng cự, điểm vào tốt, mục tiêu giá ngắn/dài hạn...",
  "market_summary": "Tổng quan thị trường 3-4 câu: tác động VN-Index, ngành, vĩ mô trong nước và quốc tế...",
  "ai_recommendation": "Khuyến nghị đầu tư chi tiết 5-6 câu: lý do cụ thể, chiến lược mua/bán/giữ, phân bổ tỷ trọng, điểm vào/ra, rủi ro chính...",
  "catalysts": ["Động lực tăng 1", "Động lực tăng 2", "Động lực tăng 3"],
  "risks": ["Rủi ro 1", "Rủi ro 2", "Rủi ro 3"],
  "signals": [
    {"label": "Mô tả tín hiệu tích cực 1", "type": "positive"},
    {"label": "Mô tả tín hiệu tích cực 2", "type": "positive"},
    {"label": "Mô tả tín hiệu tích cực 3", "type": "positive"},
    {"label": "Mô tả cần chú ý", "type": "neutral"},
    {"label": "Mô tả rủi ro 1", "type": "negative"},
    {"label": "Mô tả rủi ro 2", "type": "negative"}
  ]
}`;

  try {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: "API key chưa được cấu hình." });
    }

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 2000,
        messages: [{ role: "user", content: prompt }],
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error("Anthropic API error:", errText);
      return res.status(502).json({ error: "Lỗi kết nối AI. Vui lòng thử lại." });
    }

    const data = await response.json();
    const text = data.content.map((i) => i.text || "").join("\n");

    let result;
    try {
      const clean = text.replace(/```json|```/g, "").trim();
      result = JSON.parse(clean);
    } catch (e) {
      return res.status(200).json({ raw: text });
    }

    res.json({ success: true, data: result, ticker: cleanTicker });
  } catch (err) {
    console.error("Server error:", err);
    res.status(500).json({ error: "Lỗi máy chủ. Vui lòng thử lại." });
  }
});

// ─── API: Market overview ──────────────────────────────────────────────────
app.post("/api/market", async (req, res) => {
  const prompt = `Bạn là chuyên gia phân tích thị trường chứng khoán Việt Nam.

Hãy phân tích toàn diện thị trường chứng khoán Việt Nam hiện tại và trả về JSON (không markdown):
{
  "vnindex": { "value": "X.XXX.XX", "change": "+/-X.XX", "change_pct": "+/-X.XX%", "trend": "Tăng/Giảm/Đi ngang" },
  "hnx": { "value": "XXX.XX", "change": "+/-X.XX", "change_pct": "+/-X.XX%", "trend": "Tăng/Giảm/Đi ngang" },
  "upcom": { "value": "XXX.XX", "change": "+/-X.XX", "change_pct": "+/-X.XX%", "trend": "Tăng/Giảm/Đi ngang" },
  "market_sentiment": "Tích cực/Trung lập/Tiêu cực",
  "market_phase": "Mô tả giai đoạn thị trường hiện tại",
  "hot_sectors": ["Ngành nóng 1", "Ngành nóng 2", "Ngành nóng 3"],
  "top_picks": ["Mã 1", "Mã 2", "Mã 3", "Mã 4", "Mã 5"],
  "avoid_list": ["Mã cần tránh 1", "Mã cần tránh 2"],
  "macro_factors": "Phân tích các yếu tố vĩ mô ảnh hưởng đến thị trường 4-5 câu",
  "outlook_1m": "Nhận định xu hướng 1 tháng tới",
  "outlook_3m": "Nhận định xu hướng 3 tháng tới",
  "strategy": "Chiến lược đầu tư được khuyến nghị 3-4 câu",
  "note": "Lưu ý: số liệu chỉ số là ước tính, cần cập nhật thời gian thực"
}`;

  try {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) return res.status(500).json({ error: "API key chưa được cấu hình." });

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1500,
        messages: [{ role: "user", content: prompt }],
      }),
    });

    const data = await response.json();
    const text = data.content.map((i) => i.text || "").join("\n");
    const clean = text.replace(/```json|```/g, "").trim();
    const result = JSON.parse(clean);
    res.json({ success: true, data: result });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Lỗi phân tích thị trường." });
  }
});

// ─── Health check ──────────────────────────────────────────────────────────
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// ─── Catch-all: serve frontend ─────────────────────────────────────────────
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(PORT, () => {
  console.log(`✅  VN Stock AI Analyzer chạy tại http://localhost:${PORT}`);
});
