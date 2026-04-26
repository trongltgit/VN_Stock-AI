# VN Stock AI Analyzer 🇻🇳📈

Hệ thống phân tích chứng khoán Việt Nam bằng AI — cổ phiếu, chứng chỉ quỹ, ETF — với phân tích cơ bản, kỹ thuật, đồ thị, giá và khuyến nghị MUA/BÁN/GIỮ.

## Tính năng

- **Phân tích mã** — nhập bất kỳ mã cổ phiếu (VCB, TCB, HPG...) hoặc chứng chỉ quỹ (VCBS, MGF, VCBF-BCF...)
- **Phân tích toàn diện** — cơ bản, kỹ thuật, đồ thị, giá, thị trường
- **Khuyến nghị AI** — MUA / BÁN / GIỮ kèm mục tiêu giá, cắt lỗ, độ tin cậy
- **Tổng quan thị trường** — VN-Index, HNX, UPCOM, ngành nóng, chiến lược
- **So sánh mã** — phân tích song song tối đa 4 mã
- **Quản lý nguồn dữ liệu** — thêm/tắt website tham chiếu (VCBS, VCBF, CafeF, VietStock...)
- **Lịch sử phân tích** — xem lại các kết quả trong phiên

---

## Cấu trúc dự án

```
vn-stock-ai/
├── server.js           # Express server + Anthropic API proxy
├── package.json
├── .env.example        # Template biến môi trường
├── .gitignore
├── README.md
└── public/
    ├── index.html      # Giao diện chính
    ├── style.css       # CSS
    └── app.js          # Frontend logic
```

---

## Chạy local

### 1. Clone & cài dependencies

```bash
git clone https://github.com/YOUR_USERNAME/vn-stock-ai.git
cd vn-stock-ai
npm install
```

### 2. Tạo file `.env`

```bash
cp .env.example .env
```

Mở `.env` và điền API key:

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
PORT=3000
```

> **Lấy API key tại:** https://console.anthropic.com/settings/keys

### 3. Chạy server

```bash
npm start
# hoặc development mode (auto-reload):
npm run dev
```

Mở trình duyệt: **http://localhost:3000**

---

## Deploy lên Render (free)

### Bước 1 — Push lên GitHub

```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/vn-stock-ai.git
git push -u origin main
```

### Bước 2 — Tạo Web Service trên Render

1. Đăng nhập https://render.com
2. Click **"New +"** → **"Web Service"**
3. Kết nối GitHub repo `vn-stock-ai`
4. Điền thông tin:

| Trường | Giá trị |
|--------|---------|
| **Name** | `vn-stock-ai` (hoặc tên bạn muốn) |
| **Region** | Singapore (gần VN nhất) |
| **Branch** | `main` |
| **Runtime** | `Node` |
| **Build Command** | `npm install` |
| **Start Command** | `npm start` |
| **Plan** | `Free` |

### Bước 3 — Thêm biến môi trường

Trong Render dashboard → tab **Environment** → Add:

```
ANTHROPIC_API_KEY = sk-ant-xxxxxxxxxxxx
```

### Bước 4 — Deploy

Click **"Create Web Service"**. Render sẽ tự động build & deploy.  
URL sẽ có dạng: `https://vn-stock-ai.onrender.com`

> ⚠️ **Lưu ý Free tier:** Render free sẽ sleep sau 15 phút không có request.  
> Lần đầu vào sau khi sleep sẽ mất ~30-60 giây để wake up.

---

## Cập nhật sau khi deploy

Mỗi lần push code mới lên GitHub, Render sẽ tự động re-deploy:

```bash
git add .
git commit -m "cập nhật tính năng X"
git push
```

---

## API Endpoints

| Method | Path | Mô tả |
|--------|------|-------|
| `POST` | `/api/analyze` | Phân tích một mã chứng khoán |
| `POST` | `/api/market` | Phân tích tổng quan thị trường |
| `GET`  | `/api/health` | Kiểm tra trạng thái server |

### POST /api/analyze

```json
{
  "ticker": "VCB",
  "assetType": "stock",
  "analysisTypes": ["Phân tích cơ bản", "Phân tích kỹ thuật"],
  "sources": ["VCBS Quotes", "CafeF", "VietStock"]
}
```

---

## Tùy chỉnh thêm

### Thêm nguồn dữ liệu trong code (`public/app.js`)

```javascript
state.sources.push({
  name: 'Tên nguồn',
  url: 'https://example.com',
  type: 'data', // data | news | broker | fund | research | custom
  active: true
});
```

### Thay đổi model AI (`server.js`)

Tìm dòng:
```javascript
model: "claude-sonnet-4-20250514",
```
Thay bằng model khác nếu cần.

---

## Lưu ý pháp lý

> Kết quả phân tích từ AI **chỉ mang tính tham khảo**, không phải tư vấn đầu tư chính thức.  
> Nhà đầu tư tự chịu trách nhiệm về các quyết định đầu tư của mình.

---

## License

MIT
