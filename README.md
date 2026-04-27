# 🚀 VN Stock AI — Hướng Dẫn Deploy (100% Miễn Phí)

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────┐
│                   MULTI-AGENT WORKFLOW               │
├──────────────┬──────────────────┬───────────────────┤
│  AGENT 1     │    AGENT 2       │    AGENT 3        │
│  DuckDuckGo  │  Gemini 1.5Flash │  DeepSeek-R1      │
│  (News)      │  (Docs/PDF)      │  via Groq API     │
│  FREE        │  FREE (15 RPM)   │  FREE             │
└──────────────┴──────────────────┴───────────────────┘
```

## 📁 Cấu Trúc Files

```
vn-stock-ai/
├── backend/
│   ├── app.py              ← Flask server + 3 AI agents
│   ├── requirements.txt    ← Python dependencies
│   └── .env.example        ← Template cấu hình API keys
├── frontend/
│   └── templates/
│       └── index.html      ← Giao diện web (served by Flask)
├── render.yaml             ← Config deploy Render.com
├── Procfile                ← Config gunicorn
└── README.md               ← File này
```

## 🔑 Bước 1: Lấy API Keys (Miễn Phí)

### Groq API (DeepSeek-R1 - MIỄN PHÍ)
1. Vào https://console.groq.com
2. Đăng ký tài khoản miễn phí
3. Tạo API Key
4. Model sử dụng: `deepseek-r1-distill-llama-70b` (free tier)

### Google Gemini API (MIỄN PHÍ)
1. Vào https://aistudio.google.com
2. Đăng nhập bằng Google account
3. Tạo API Key
4. Model: `gemini-1.5-flash` (miễn phí 15 requests/phút)

## 🌐 Bước 2: Deploy lên Render.com (MIỄN PHÍ)

### Cách 1: Deploy qua GitHub (Khuyên dùng)
1. Tạo repo GitHub mới
2. Push toàn bộ code lên
3. Vào https://render.com → New → Web Service
4. Kết nối GitHub repo
5. Cấu hình:
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `gunicorn backend.app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
6. Thêm Environment Variables:
   - `GROQ_API_KEY` = key của bạn
   - `GEMINI_API_KEY` = key của bạn
7. Chọn **Free plan** → Deploy!

### Cách 2: Deploy Local (Test)
```bash
cd vn-stock-ai
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
# Sửa .env với API keys thực của bạn
cd backend
python app.py
# Mở http://localhost:5000
```

## 🖥️ Giao Diện Web

- **Tab Cổ Phiếu**: Nhập mã như VCB, HPG, FPT...
- **Tab Chứng Chỉ Quỹ**: Nhập mã như MAFPF1, VFMVSF...
- **Tab Ngoại Tệ**: Nhập cặp như USD.VND, EUR.USD, USD.JPY...
- **URL Sources**: Paste URL từ VCBS, VCBF để lấy thêm dữ liệu
- **Upload PDF**: Upload BCTC, prospectus để phân tích sâu hơn

## ⚡ Tính Năng

✅ Phân tích cơ bản (tài chính, định giá, rủi ro)
✅ Phân tích kỹ thuật (xu hướng, hỗ trợ/kháng cự, RSI, MACD)
✅ Đọc và phân tích báo cáo tài chính PDF
✅ Quét tin tức thị trường real-time
✅ Phân tích tỷ giá ngoại tệ
✅ Khuyến nghị MUA/BÁN/GIỮ có luận cứ
✅ Giao diện professional dark theme
✅ Ticker tape giá realtime
✅ Hoàn toàn miễn phí!

## 🔧 Models AI Sử Dụng (Tất Cả Miễn Phí)

| Agent | Model | Provider | Giới hạn Free |
|-------|-------|----------|---------------|
| News | DuckDuckGo | N/A | Unlimited |
| Document | gemini-1.5-flash | Google AI Studio | 15 RPM, 1M TPM |
| Reasoning | deepseek-r1-distill-llama-70b | Groq | 30 RPM, 14400 RPD |
| Fallback | llama-3.3-70b-versatile | Groq | 30 RPM |

## ⚠️ Lưu Ý

- Render Free tier có thể sleep sau 15 phút không dùng
- Lần đầu load có thể mất 30-60 giây để warm up
- PDF tối đa 5 files, mỗi file không quá 10MB
- Phân tích mỗi lần mất khoảng 15-30 giây
