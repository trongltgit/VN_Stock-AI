  """
VN Stock AI — Multi-Agent Financial Analysis
Python 3.11 | Flask | Groq DeepSeek-R1 | Gemini | DuckDuckGo
"""

import os, uuid, tempfile, logging
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

INDEX_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>VN Stock AI — Phân Tích Chứng Khoán Chuyên Sâu</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg:        #060b14;
  --bg2:       #0c1421;
  --bg3:       #111d2e;
  --border:    #1e3050;
  --border2:   #2a4570;
  --accent:    #00d4ff;
  --accent2:   #0099cc;
  --gold:      #f0c040;
  --green:     #00e676;
  --red:       #ff5252;
  --yellow:    #ffd740;
  --text:      #e8f4fd;
  --text2:     #8baabb;
  --text3:     #4a6b88;
  --card-bg:   rgba(12,20,33,0.95);
  --glow:      0 0 30px rgba(0,212,255,0.15);
  --glow2:     0 0 60px rgba(0,212,255,0.08);
}

*{margin:0;padding:0;box-sizing:border-box;}

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', sans-serif;
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── BACKGROUND GRID ── */
body::before {
  content:'';
  position:fixed;
  inset:0;
  background-image:
    linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events:none;
  z-index:0;
}

/* ── SCANLINE OVERLAY ── */
body::after {
  content:'';
  position:fixed;
  inset:0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.08) 2px,
    rgba(0,0,0,0.08) 4px
  );
  pointer-events:none;
  z-index:0;
}

/* ── HEADER ── */
header {
  position: relative;
  z-index:10;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 2rem;
  height: 64px;
  border-bottom: 1px solid var(--border);
  background: rgba(6,11,20,0.95);
  backdrop-filter: blur(12px);
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-icon {
  width: 36px;
  height: 36px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  box-shadow: 0 0 16px rgba(0,212,255,0.4);
}

.logo-text {
  font-family: 'Syne', sans-serif;
  font-size: 1.2rem;
  font-weight: 800;
  letter-spacing: -0.02em;
}

.logo-text span { color: var(--accent); }

.header-status {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  font-size: 0.75rem;
  font-family: 'Space Mono', monospace;
  color: var(--text3);
}

.status-dot {
  display: flex;
  align-items: center;
  gap: 6px;
}

.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--green);
  animation: blink 2s infinite;
}

@keyframes blink {
  0%,100%{opacity:1;} 50%{opacity:0.3;}
}

/* ── TICKER TAPE ── */
.ticker-wrap {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 6px 0;
  overflow: hidden;
  position: relative;
  z-index: 9;
}

.ticker {
  display: flex;
  gap: 3rem;
  animation: scroll-ticker 40s linear infinite;
  white-space: nowrap;
}

@keyframes scroll-ticker {
  0%{transform:translateX(0);}
  100%{transform:translateX(-50%);}
}

.tick-item {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: 'Space Mono', monospace;
  font-size: 0.72rem;
}

.tick-sym { color: var(--accent); font-weight: 700; }
.tick-val { color: var(--text); }
.tick-up  { color: var(--green); }
.tick-dn  { color: var(--red); }

/* ── MAIN LAYOUT ── */
main {
  position: relative;
  z-index: 1;
  max-width: 1400px;
  margin: 0 auto;
  padding: 2rem 1.5rem;
  display: grid;
  grid-template-columns: 380px 1fr;
  gap: 1.5rem;
  align-items: start;
}

/* ── SIDEBAR ── */
.sidebar {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

/* ── CARDS ── */
.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.card:hover {
  border-color: var(--border2);
  box-shadow: var(--glow);
}

.card::before {
  content:'';
  position:absolute;
  top:0; left:0; right:0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  opacity: 0.5;
}

.card-title {
  font-family: 'Syne', sans-serif;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* ── TABS ── */
.tabs {
  display: flex;
  gap: 4px;
  background: var(--bg);
  border-radius: 8px;
  padding: 3px;
  margin-bottom: 1rem;
}

.tab {
  flex: 1;
  padding: 7px 10px;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: var(--text2);
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem;
  cursor: pointer;
  transition: all 0.2s;
  text-align: center;
}

.tab.active {
  background: var(--accent);
  color: var(--bg);
  font-weight: 700;
}

.tab:hover:not(.active) {
  background: var(--border);
  color: var(--text);
}

/* ── FORM ELEMENTS ── */
.form-group {
  margin-bottom: 0.85rem;
}

label {
  display: block;
  font-size: 0.7rem;
  font-family: 'Space Mono', monospace;
  color: var(--text3);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 5px;
}

input[type="text"], input[type="url"], select {
  width: 100%;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 9px 12px;
  color: var(--text);
  font-family: 'Space Mono', monospace;
  font-size: 0.82rem;
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
}

input[type="text"]:focus, input[type="url"]:focus, select:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(0,212,255,0.1);
}

input::placeholder { color: var(--text3); }

.input-group {
  display: flex;
  gap: 6px;
}

.input-group input { flex:1; }

.tag-btn {
  padding: 9px 10px;
  background: var(--border);
  border: 1px solid var(--border2);
  border-radius: 8px;
  color: var(--accent);
  font-size: 0.7rem;
  font-family: 'Space Mono', monospace;
  cursor: pointer;
  white-space: nowrap;
  transition: background 0.2s;
}

.tag-btn:hover { background: var(--border2); }

/* ── URL INPUTS ── */
.url-list { display: flex; flex-direction: column; gap: 6px; }

.url-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.url-item input { flex:1; font-size: 0.72rem; }

.url-del {
  background: none;
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--red);
  width: 28px;
  height: 28px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  transition: background 0.2s;
}

.url-del:hover { background: rgba(255,82,82,0.1); }

.add-url-btn {
  width: 100%;
  padding: 7px;
  background: none;
  border: 1px dashed var(--border2);
  border-radius: 8px;
  color: var(--text3);
  font-size: 0.72rem;
  font-family: 'Space Mono', monospace;
  cursor: pointer;
  margin-top: 6px;
  transition: all 0.2s;
}

.add-url-btn:hover { border-color: var(--accent); color: var(--accent); }

/* ── FILE UPLOAD ── */
.upload-zone {
  border: 1px dashed var(--border2);
  border-radius: 8px;
  padding: 1rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
}

.upload-zone:hover, .upload-zone.drag-over {
  border-color: var(--accent);
  background: rgba(0,212,255,0.03);
}

.upload-zone input { position:absolute; inset:0; opacity:0; cursor:pointer; }
.upload-zone .upload-icon { font-size: 1.5rem; margin-bottom: 4px; }
.upload-zone p { font-size: 0.72rem; color: var(--text3); }
.upload-zone span { color: var(--accent); }

.file-list { margin-top: 8px; display: flex; flex-direction: column; gap: 4px; }

.file-chip {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 10px;
  font-size: 0.7rem;
  font-family: 'Space Mono', monospace;
  color: var(--accent);
}

.file-chip button {
  background: none;
  border: none;
  color: var(--red);
  cursor: pointer;
  font-size: 0.75rem;
}

/* ── ANALYZE BUTTON ── */
.btn-analyze {
  width: 100%;
  padding: 13px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  border: none;
  border-radius: 10px;
  color: var(--bg);
  font-family: 'Syne', sans-serif;
  font-size: 0.9rem;
  font-weight: 800;
  letter-spacing: 0.05em;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  box-shadow: 0 4px 20px rgba(0,212,255,0.3);
}

.btn-analyze:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 28px rgba(0,212,255,0.45);
}

.btn-analyze:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}

/* ── QUICK SYMBOLS ── */
.quick-symbols {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-top: 6px;
}

.sym-chip {
  padding: 4px 10px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 20px;
  font-size: 0.68rem;
  font-family: 'Space Mono', monospace;
  color: var(--text2);
  cursor: pointer;
  transition: all 0.2s;
}

.sym-chip:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: rgba(0,212,255,0.05);
}

/* ── CONTENT AREA ── */
.content-area {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  min-height: 600px;
}

/* ── WELCOME STATE ── */
.welcome-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 500px;
  text-align: center;
  gap: 1.5rem;
}

.welcome-logo {
  font-size: 3rem;
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%,100%{transform:translateY(0);}
  50%{transform:translateY(-10px);}
}

.welcome-state h2 {
  font-family: 'Syne', sans-serif;
  font-size: 1.8rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--accent), var(--gold));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.welcome-state p { color: var(--text2); font-size: 0.9rem; max-width: 400px; }

.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  width: 100%;
  max-width: 600px;
}

.feature-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem;
  text-align: center;
}

.feature-card .fi { font-size: 1.5rem; margin-bottom: 6px; }
.feature-card h4 { font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 700; color: var(--accent); margin-bottom: 4px; }
.feature-card p { font-size: 0.68rem; color: var(--text3); }

/* ── LOADING STATE ── */
.loading-state {
  display: none;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 500px;
  gap: 2rem;
}

.loading-state.active { display: flex; }

.spinner-container { position: relative; }

.spinner-ring {
  width: 80px;
  height: 80px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.spinner-inner {
  position: absolute;
  inset: 10px;
  border: 2px solid var(--border);
  border-bottom-color: var(--gold);
  border-radius: 50%;
  animation: spin 0.6s linear infinite reverse;
}

.loading-steps {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  width: 100%;
  max-width: 360px;
}

.step {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px;
  border-radius: 8px;
  font-size: 0.78rem;
  font-family: 'Space Mono', monospace;
  color: var(--text3);
  background: var(--bg2);
  border: 1px solid var(--border);
  transition: all 0.3s;
}

.step.active {
  color: var(--accent);
  border-color: var(--accent);
  background: rgba(0,212,255,0.05);
}

.step.done {
  color: var(--green);
  border-color: var(--green);
  background: rgba(0,230,118,0.05);
}

.step-icon { font-size: 1rem; }

/* ── RESULT AREA ── */
.result-area { display: none; flex-direction: column; gap: 1rem; }
.result-area.active { display: flex; }

/* ── RESULT HEADER ── */
.result-header {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 1rem;
}

.result-header::before {
  content:'';
  position:absolute;
  top:0; left:0; right:0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--gold), transparent);
  border-radius: 12px 12px 0 0;
}

.result-header { position: relative; }

.sym-display {
  display: flex;
  align-items: center;
  gap: 16px;
}

.sym-code {
  font-family: 'Syne', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  color: var(--accent);
  letter-spacing: -0.02em;
}

.sym-meta { display: flex; flex-direction: column; gap: 2px; }
.sym-type {
  font-size: 0.65rem;
  font-family: 'Space Mono', monospace;
  color: var(--text3);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.sym-time { font-size: 0.7rem; color: var(--text3); font-family: 'Space Mono', monospace; }

.rec-badge {
  padding: 8px 20px;
  border-radius: 8px;
  font-family: 'Syne', sans-serif;
  font-size: 1rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  display: flex;
  align-items: center;
  gap: 8px;
}

.rec-BUY   { background: rgba(0,230,118,0.15); border: 2px solid var(--green); color: var(--green); }
.rec-SELL  { background: rgba(255,82,82,0.15);  border: 2px solid var(--red);   color: var(--red); }
.rec-HOLD  { background: rgba(240,192,64,0.15); border: 2px solid var(--gold);  color: var(--gold); }
.rec-WATCH { background: rgba(0,212,255,0.1);   border: 2px solid var(--accent);color: var(--accent); }

/* ── AGENT BADGES ── */
.agent-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.agent-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 10px;
  border-radius: 6px;
  font-size: 0.68rem;
  font-family: 'Space Mono', monospace;
  border: 1px solid;
}

.agent-badge.ok    { border-color: var(--green); color: var(--green); background: rgba(0,230,118,0.05); }
.agent-badge.warn  { border-color: var(--yellow); color: var(--yellow); background: rgba(255,215,64,0.05); }

/* ── ANALYSIS CONTENT ── */
.analysis-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}

.analysis-card-header {
  padding: 0.85rem 1.25rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: rgba(0,0,0,0.2);
}

.analysis-card-header h3 {
  font-family: 'Syne', sans-serif;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--accent);
  display: flex;
  align-items: center;
  gap: 8px;
}

.analysis-body {
  padding: 1.5rem;
  line-height: 1.8;
  font-size: 0.875rem;
  color: var(--text);
}

/* ── MARKDOWN-LIKE STYLING for AI output ── */
.analysis-body h1, .analysis-body h2, .analysis-body h3 {
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  color: var(--accent);
  margin: 1.2rem 0 0.5rem;
}

.analysis-body h2 { font-size: 1rem; color: var(--gold); border-bottom: 1px solid var(--border); padding-bottom: 6px; }
.analysis-body h3 { font-size: 0.9rem; color: var(--accent); }

.analysis-body strong { color: var(--text); font-weight: 600; }
.analysis-body em { color: var(--text2); font-style: italic; }

.analysis-body ul, .analysis-body ol {
  padding-left: 1.5rem;
  margin: 0.5rem 0;
}

.analysis-body li { margin-bottom: 4px; }

.analysis-body hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1rem 0;
}

.analysis-body blockquote {
  border-left: 3px solid var(--accent);
  padding: 8px 16px;
  background: rgba(0,212,255,0.05);
  border-radius: 0 8px 8px 0;
  margin: 0.8rem 0;
  color: var(--text2);
}

/* ── COPY BTN ── */
.copy-btn {
  padding: 5px 12px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text2);
  font-size: 0.68rem;
  font-family: 'Space Mono', monospace;
  cursor: pointer;
  transition: all 0.2s;
}

.copy-btn:hover { border-color: var(--accent); color: var(--accent); }

/* ── CHART PLACEHOLDER ── */
.chart-placeholder {
  background: var(--bg2);
  border: 1px dashed var(--border2);
  border-radius: 8px;
  padding: 3rem;
  text-align: center;
  color: var(--text3);
  font-size: 0.78rem;
  font-family: 'Space Mono', monospace;
}

/* ── ERROR STATE ── */
.error-box {
  background: rgba(255,82,82,0.08);
  border: 1px solid var(--red);
  border-radius: 10px;
  padding: 1rem 1.25rem;
  color: var(--red);
  font-size: 0.82rem;
  display: none;
}

.error-box.active { display: block; }

/* ── RESPONSIVE ── */
@media (max-width: 900px) {
  main { grid-template-columns: 1fr; }
  .feature-grid { grid-template-columns: 1fr 1fr; }
}

@media (max-width: 540px) {
  header { padding: 0 1rem; }
  main { padding: 1rem; }
  .feature-grid { grid-template-columns: 1fr; }
  .sym-code { font-size: 1.4rem; }
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* ── TOOLTIP ── */
.tooltip {
  position: relative;
}
.tooltip::after {
  content: attr(data-tip);
  position: absolute;
  bottom: calc(100% + 6px);
  left: 50%;
  transform: translateX(-50%);
  background: var(--bg3);
  border: 1px solid var(--border2);
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 0.65rem;
  font-family: 'Space Mono', monospace;
  color: var(--text2);
  white-space: nowrap;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.2s;
  z-index: 100;
}
.tooltip:hover::after { opacity: 1; }

/* ── FOREX DIRECTION ── */
.dir-UP       { color: var(--green); }
.dir-DOWN     { color: var(--red); }
.dir-SIDEWAYS { color: var(--yellow); }
</style>
</head>
<body>

<!-- HEADER -->
<header>
  <div class="logo">
    <div class="logo-icon">📈</div>
    <div>
      <div class="logo-text">VN<span>Stock</span>AI</div>
    </div>
  </div>
  <div class="header-status">
    <div class="status-dot">
      <div class="dot"></div>
      <span id="systemStatus">Đang kết nối...</span>
    </div>
    <span id="currentTime" style="color:var(--text2)"></span>
  </div>
</header>

<!-- TICKER TAPE -->
<div class="ticker-wrap">
  <div class="ticker" id="tickerTape">
    <span class="tick-item"><span class="tick-sym">VN-INDEX</span><span class="tick-val">1,287.45</span><span class="tick-up">▲ +12.3 (+0.97%)</span></span>
    <span class="tick-item"><span class="tick-sym">VCB</span><span class="tick-val">86,500</span><span class="tick-up">▲ +500</span></span>
    <span class="tick-item"><span class="tick-sym">VHM</span><span class="tick-val">38,200</span><span class="tick-dn">▼ -300</span></span>
    <span class="tick-item"><span class="tick-sym">HPG</span><span class="tick-val">23,100</span><span class="tick-up">▲ +200</span></span>
    <span class="tick-item"><span class="tick-sym">MAFPF1</span><span class="tick-val">12,540</span><span class="tick-up">▲ +85</span></span>
    <span class="tick-item"><span class="tick-sym">USD/VND</span><span class="tick-val">25,480</span><span class="tick-dn">▼ -20</span></span>
    <span class="tick-item"><span class="tick-sym">VIC</span><span class="tick-val">41,500</span><span class="tick-up">▲ +350</span></span>
    <span class="tick-item"><span class="tick-sym">MBB</span><span class="tick-val">19,800</span><span class="tick-dn">▼ -150</span></span>
    <span class="tick-item"><span class="tick-sym">FPT</span><span class="tick-val">113,600</span><span class="tick-up">▲ +1,200</span></span>
    <span class="tick-item"><span class="tick-sym">TCB</span><span class="tick-val">22,400</span><span class="tick-up">▲ +100</span></span>
    <!-- Duplicate for seamless loop -->
    <span class="tick-item"><span class="tick-sym">VN-INDEX</span><span class="tick-val">1,287.45</span><span class="tick-up">▲ +12.3 (+0.97%)</span></span>
    <span class="tick-item"><span class="tick-sym">VCB</span><span class="tick-val">86,500</span><span class="tick-up">▲ +500</span></span>
    <span class="tick-item"><span class="tick-sym">VHM</span><span class="tick-val">38,200</span><span class="tick-dn">▼ -300</span></span>
    <span class="tick-item"><span class="tick-sym">HPG</span><span class="tick-val">23,100</span><span class="tick-up">▲ +200</span></span>
    <span class="tick-item"><span class="tick-sym">MAFPF1</span><span class="tick-val">12,540</span><span class="tick-up">▲ +85</span></span>
    <span class="tick-item"><span class="tick-sym">USD/VND</span><span class="tick-val">25,480</span><span class="tick-dn">▼ -20</span></span>
    <span class="tick-item"><span class="tick-sym">VIC</span><span class="tick-val">41,500</span><span class="tick-up">▲ +350</span></span>
    <span class="tick-item"><span class="tick-sym">MBB</span><span class="tick-val">19,800</span><span class="tick-dn">▼ -150</span></span>
    <span class="tick-item"><span class="tick-sym">FPT</span><span class="tick-val">113,600</span><span class="tick-up">▲ +1,200</span></span>
    <span class="tick-item"><span class="tick-sym">TCB</span><span class="tick-val">22,400</span><span class="tick-up">▲ +100</span></span>
  </div>
</div>

<!-- MAIN -->
<main>
  <!-- SIDEBAR -->
  <aside class="sidebar">

    <!-- INPUT CARD -->
    <div class="card">
      <div class="card-title">⚙ Cấu Hình Phân Tích</div>

      <!-- TABS: Stock / Fund / Forex -->
      <div class="tabs">
        <button class="tab active" data-mode="stock" onclick="setMode('stock',this)">📊 Cổ phiếu</button>
        <button class="tab" data-mode="fund"  onclick="setMode('fund',this)">🏦 Quỹ</button>
        <button class="tab" data-mode="forex" onclick="setMode('forex',this)">💱 Ngoại tệ</button>
      </div>

      <!-- SYMBOL INPUT -->
      <div class="form-group">
        <label id="symbolLabel">🔤 Mã Chứng Khoán</label>
        <div class="input-group">
          <input type="text" id="symbolInput" placeholder="VD: VCB, HPG, FPT..." style="text-transform:uppercase"/>
          <button class="tag-btn" onclick="analyzeNow()">▶ Run</button>
        </div>
        <div class="quick-symbols" id="quickSymbols">
          <span class="sym-chip" onclick="fillSym('VCB')">VCB</span>
          <span class="sym-chip" onclick="fillSym('VHM')">VHM</span>
          <span class="sym-chip" onclick="fillSym('HPG')">HPG</span>
          <span class="sym-chip" onclick="fillSym('FPT')">FPT</span>
          <span class="sym-chip" onclick="fillSym('TCB')">TCB</span>
          <span class="sym-chip" onclick="fillSym('MBB')">MBB</span>
        </div>
      </div>

      <!-- URL SOURCES -->
      <div class="form-group" id="urlSection">
        <label>🔗 Nguồn Dữ Liệu URL (Tùy chọn)</label>
        <div class="url-list" id="urlList">
          <div class="url-item">
            <input type="url" placeholder="https://quotes.vcbs.com.vn/k8/"/>
            <button class="url-del" onclick="removeUrl(this)">✕</button>
          </div>
          <div class="url-item">
            <input type="url" placeholder="https://www.vcbf.com/"/>
            <button class="url-del" onclick="removeUrl(this)">✕</button>
          </div>
        </div>
        <button class="add-url-btn" onclick="addUrlInput()">＋ Thêm URL</button>
      </div>

      <!-- PDF UPLOAD (only for stock/fund) -->
      <div class="form-group" id="uploadSection">
        <label>📄 Upload Báo Cáo Tài Chính (PDF)</label>
        <div class="upload-zone" id="uploadZone">
          <input type="file" id="pdfInput" multiple accept=".pdf" onchange="handleFiles(this.files)"/>
          <div class="upload-icon">📂</div>
          <p><span>Chọn file</span> hoặc kéo thả vào đây</p>
          <p style="margin-top:4px;font-size:0.65rem;">Hỗ trợ: .pdf (tối đa 5 files)</p>
        </div>
        <div class="file-list" id="fileList"></div>
      </div>

      <!-- ANALYZE BUTTON -->
      <button class="btn-analyze" id="analyzeBtn" onclick="analyzeNow()">
        <span>🤖</span> Phân Tích AI Chuyên Sâu
      </button>
    </div>

    <!-- AGENT STATUS CARD -->
    <div class="card">
      <div class="card-title">🧬 Trạng Thái AI Agents</div>
      <div style="display:flex;flex-direction:column;gap:8px;" id="agentStatus">
        <div class="step" id="ag1">
          <span class="step-icon">🔍</span>
          <div>
            <div style="font-size:0.72rem;color:var(--text2)">Agent 1: News Collector</div>
            <div style="font-size:0.65rem;color:var(--text3)">DuckDuckGo Search</div>
          </div>
        </div>
        <div class="step" id="ag2">
          <span class="step-icon">📖</span>
          <div>
            <div style="font-size:0.72rem;color:var(--text2)">Agent 2: Doc Analyzer</div>
            <div style="font-size:0.65rem;color:var(--text3)">Gemini 1.5 Flash (Free)</div>
          </div>
        </div>
        <div class="step" id="ag3">
          <span class="step-icon">🧠</span>
          <div>
            <div style="font-size:0.72rem;color:var(--text2)">Agent 3: Reasoning</div>
            <div style="font-size:0.65rem;color:var(--text3)">DeepSeek-R1 via Groq (Free)</div>
          </div>
        </div>
      </div>
    </div>

    <!-- API CONFIG HINT -->
    <div class="card" style="border-color:rgba(240,192,64,0.3)">
      <div class="card-title" style="color:var(--gold)">🔑 Cấu Hình API Keys</div>
      <p style="font-size:0.72rem;color:var(--text3);line-height:1.7;">
        Thêm vào file <code style="color:var(--accent)">.env</code> trên server:<br/>
        <code style="color:var(--green)">GROQ_API_KEY</code> → <a href="https://console.groq.com" target="_blank" style="color:var(--accent)">console.groq.com</a><br/>
        <code style="color:var(--green)">GEMINI_API_KEY</code> → <a href="https://aistudio.google.com" target="_blank" style="color:var(--accent)">aistudio.google.com</a><br/>
        <br/>
        <span style="color:var(--yellow)">⚡ Cả hai đều hoàn toàn miễn phí!</span>
      </p>
    </div>

  </aside>

  <!-- CONTENT AREA -->
  <div class="content-area">

    <!-- WELCOME -->
    <div class="welcome-state" id="welcomeState">
      <div class="welcome-logo">📊</div>
      <h2>Phân Tích Chứng Khoán AI</h2>
      <p>Hệ thống multi-agent phân tích chuyên sâu cổ phiếu, chứng chỉ quỹ và tỷ giá ngoại tệ cho thị trường Việt Nam</p>
      <div class="feature-grid">
        <div class="feature-card">
          <div class="fi">🔍</div>
          <h4>Tin Tức Thị Trường</h4>
          <p>DuckDuckGo quét tin tức vĩ mô real-time</p>
        </div>
        <div class="feature-card">
          <div class="fi">📄</div>
          <h4>Đọc Báo Cáo PDF</h4>
          <p>Gemini 1.5 Flash phân tích BCTC chuyên sâu</p>
        </div>
        <div class="feature-card">
          <div class="fi">🧠</div>
          <h4>Suy Luận Định Giá</h4>
          <p>DeepSeek-R1 đưa ra khuyến nghị MUA/BÁN/GIỮ</p>
        </div>
      </div>
    </div>

    <!-- LOADING -->
    <div class="loading-state" id="loadingState">
      <div class="spinner-container">
        <div class="spinner-ring"></div>
        <div class="spinner-inner"></div>
      </div>
      <div style="text-align:center">
        <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:var(--accent)" id="loadingSymbol">Đang phân tích...</div>
        <div style="font-size:0.75rem;color:var(--text3);margin-top:4px;font-family:'Space Mono',monospace">Multi-Agent AI Processing</div>
      </div>
      <div class="loading-steps">
        <div class="step" id="step1"><span class="step-icon">🔍</span> Thu thập tin tức thị trường...</div>
        <div class="step" id="step2"><span class="step-icon">📖</span> Phân tích tài liệu tài chính...</div>
        <div class="step" id="step3"><span class="step-icon">🧠</span> Suy luận & định giá chuyên sâu...</div>
        <div class="step" id="step4"><span class="step-icon">📝</span> Tổng hợp báo cáo...</div>
      </div>
    </div>

    <!-- ERROR -->
    <div class="error-box" id="errorBox"></div>

    <!-- RESULT -->
    <div class="result-area" id="resultArea">

      <!-- Result Header -->
      <div class="result-header" id="resultHeader">
        <div class="sym-display">
          <div class="sym-code" id="resultSym">--</div>
          <div class="sym-meta">
            <div class="sym-type" id="resultType">--</div>
            <div class="sym-time" id="resultTime">--</div>
          </div>
        </div>
        <div class="rec-badge" id="recBadge">—</div>
      </div>

      <!-- Agent summary row -->
      <div class="agent-row" id="agentRow"></div>

      <!-- Analysis Content -->
      <div class="analysis-card">
        <div class="analysis-card-header">
          <h3>📋 Báo Cáo Phân Tích Toàn Diện</h3>
          <button class="copy-btn" onclick="copyAnalysis()">📋 Sao chép</button>
        </div>
        <div class="analysis-body" id="analysisBody">
          <!-- AI output here -->
        </div>
      </div>

    </div>
  </div>
</main>

<script>
// ── CONFIG ──
const API_BASE = window.location.origin; // same origin (Flask serves this)
let currentMode = 'stock';
let uploadedFiles = [];

// ── TIME ──
function updateTime() {
  const now = new Date();
  document.getElementById('currentTime').textContent =
    now.toLocaleTimeString('vi-VN', {hour:'2-digit',minute:'2-digit',second:'2-digit'});
}
setInterval(updateTime, 1000);
updateTime();

// ── HEALTH CHECK ──
async function checkHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`);
    const d = await r.json();
    if (d.status === 'ok') {
      document.getElementById('systemStatus').textContent = 'Hệ thống hoạt động';
      document.getElementById('systemStatus').style.color = 'var(--green)';
      // Update agent status
      if (d.agents.news)      markAgent('ag1','done');
      if (d.agents.document)  markAgent('ag2','done');
      if (d.agents.reasoning) markAgent('ag3','done');
    }
  } catch(e) {
    document.getElementById('systemStatus').textContent = 'Chưa kết nối backend';
    document.getElementById('systemStatus').style.color = 'var(--yellow)';
  }
}
checkHealth();

function markAgent(id, state) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove('active','done');
  if (state) el.classList.add(state);
}

// ── MODE ──
const modeConfig = {
  stock: {
    label: '🔤 Mã Cổ Phiếu',
    placeholder: 'VD: VCB, HPG, FPT...',
    chips: ['VCB','VHM','HPG','FPT','TCB','MBB'],
    showUpload: true,
    showUrl: true
  },
  fund: {
    label: '🔤 Mã Chứng Chỉ Quỹ',
    placeholder: 'VD: MAFPF1, VFMVSF, SSISCA...',
    chips: ['MAFPF1','VFMVSF','SSISCA','FVBF','DCDS'],
    showUpload: true,
    showUrl: true
  },
  forex: {
    label: '💱 Cặp Tiền Tệ',
    placeholder: 'VD: USD.VND, EUR.USD, USD.JPY...',
    chips: ['USD.VND','EUR.USD','EUR.VND','USD.JPY','GBP.USD'],
    showUpload: false,
    showUrl: false
  }
};

function setMode(mode, btn) {
  currentMode = mode;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  const cfg = modeConfig[mode];
  document.getElementById('symbolLabel').textContent = cfg.label;
  document.getElementById('symbolInput').placeholder = cfg.placeholder;
  document.getElementById('symbolInput').value = '';
  // Quick chips
  const qsEl = document.getElementById('quickSymbols');
  qsEl.innerHTML = cfg.chips.map(c => `<span class="sym-chip" onclick="fillSym('${c}')">${c}</span>`).join('');
  // Show/hide sections
  document.getElementById('uploadSection').style.display = cfg.showUpload ? '' : 'none';
  document.getElementById('urlSection').style.display   = cfg.showUrl    ? '' : 'none';
}

function fillSym(sym) {
  document.getElementById('symbolInput').value = sym;
}

// ── URL MANAGEMENT ──
function addUrlInput() {
  const list = document.getElementById('urlList');
  const div = document.createElement('div');
  div.className = 'url-item';
  div.innerHTML = `<input type="url" placeholder="https://..."/><button class="url-del" onclick="removeUrl(this)">✕</button>`;
  list.appendChild(div);
}

function removeUrl(btn) {
  btn.parentElement.remove();
}

// ── FILE HANDLING ──
function handleFiles(files) {
  Array.from(files).forEach(f => {
    if (f.type === 'application/pdf' && uploadedFiles.length < 5) {
      uploadedFiles.push(f);
    }
  });
  renderFileList();
}

function renderFileList() {
  const fl = document.getElementById('fileList');
  fl.innerHTML = uploadedFiles.map((f, i) => `
    <div class="file-chip">
      <span>📄 ${f.name.substring(0,30)}${f.name.length>30?'...':''}</span>
      <button onclick="removeFile(${i})">✕</button>
    </div>
  `).join('');
}

function removeFile(i) {
  uploadedFiles.splice(i, 1);
  renderFileList();
}

// ── DRAG & DROP ──
const zone = document.getElementById('uploadZone');
zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
zone.addEventListener('drop', e => {
  e.preventDefault();
  zone.classList.remove('drag-over');
  handleFiles(e.dataTransfer.files);
});

// ── LOADING ANIMATION ──
let stepTimer = null;
function startLoadingSteps() {
  ['step1','step2','step3','step4'].forEach(s => {
    const el = document.getElementById(s);
    if(el){ el.classList.remove('active','done'); }
  });
  let i = 0;
  const steps = ['step1','step2','step3','step4'];
  stepTimer = setInterval(() => {
    if (i > 0) {
      const prev = document.getElementById(steps[i-1]);
      if(prev){ prev.classList.remove('active'); prev.classList.add('done'); }
    }
    if (i < steps.length) {
      const cur = document.getElementById(steps[i]);
      if(cur){ cur.classList.add('active'); }
      i++;
    } else {
      clearInterval(stepTimer);
    }
  }, 1800);
}

// ── ANALYZE ──
async function analyzeNow() {
  const sym = document.getElementById('symbolInput').value.trim().toUpperCase();
  if (!sym) {
    alert('Vui lòng nhập mã chứng khoán hoặc cặp tiền tệ');
    return;
  }

  // Hide others, show loading
  document.getElementById('welcomeState').style.display = 'none';
  document.getElementById('resultArea').classList.remove('active');
  document.getElementById('errorBox').classList.remove('active');
  document.getElementById('loadingState').classList.add('active');
  document.getElementById('loadingSymbol').textContent = `Đang phân tích ${sym}...`;
  document.getElementById('analyzeBtn').disabled = true;

  startLoadingSteps();

  const formData = new FormData();
  formData.append('symbol', sym);
  formData.append('type', currentMode);

  // URLs
  const urlInputs = document.querySelectorAll('#urlList input[type="url"]');
  const urls = Array.from(urlInputs).map(i => i.value.trim()).filter(Boolean);
  formData.append('urls', JSON.stringify(urls));

  // PDFs
  uploadedFiles.forEach(f => formData.append('pdfs', f));

  try {
    const resp = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      body: formData
    });

    clearInterval(stepTimer);

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.error || 'Lỗi server');
    }

    const json = await resp.json();
    renderResult(json, sym);

  } catch(e) {
    clearInterval(stepTimer);
    document.getElementById('loadingState').classList.remove('active');
    const eb = document.getElementById('errorBox');
    eb.textContent = `⚠ Lỗi: ${e.message}`;
    eb.classList.add('active');
  } finally {
    document.getElementById('analyzeBtn').disabled = false;
  }
}

// ── ENTER KEY ──
document.getElementById('symbolInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') analyzeNow();
});

// ── RENDER RESULT ──
function renderResult(json, sym) {
  document.getElementById('loadingState').classList.remove('active');

  const data = json.data;
  const mode = json.mode;

  // Header
  document.getElementById('resultSym').textContent = sym;

  const typeMap = {
    stock: 'Cổ Phiếu | HOSE/HNX',
    fund:  'Chứng Chỉ Quỹ',
    forex: 'Cặp Tiền Tệ'
  };
  document.getElementById('resultType').textContent = typeMap[currentMode] || currentMode;
  document.getElementById('resultTime').textContent =
    'Phân tích lúc: ' + new Date().toLocaleString('vi-VN');

  // Recommendation badge
  const badge = document.getElementById('recBadge');
  if (mode === 'forex') {
    const dir = data.direction || 'SIDEWAYS';
    const dirMap = { UP: '▲ TĂNG', DOWN: '▼ GIẢM', SIDEWAYS: '↔ ĐI NGANG' };
    badge.textContent = dirMap[dir] || dir;
    badge.className = `rec-badge dir-${dir}`;
  } else {
    const rec = data.recommendation || 'WATCH';
    const recMap = { BUY:'🟢 MUA', SELL:'🔴 BÁN', HOLD:'🟡 GIỮ', WATCH:'🔵 THEO DÕI' };
    badge.textContent = recMap[rec] || rec;
    badge.className = `rec-badge rec-${rec}`;
  }

  // Agent row
  const agRow = document.getElementById('agentRow');
  agRow.innerHTML = `
    <div class="agent-badge ok">🔍 DuckDuckGo: ${data.news_count||0} tin tức</div>
    <div class="agent-badge ${data.has_documents ? 'ok' : 'warn'}">📖 Gemini: ${data.has_documents ? 'Đã đọc PDF' : 'Không có PDF'}</div>
    <div class="agent-badge ok">🧠 DeepSeek-R1: Hoàn thành</div>
  `;

  // Analysis body — render markdown-ish
  const body = data.analysis || data.summary || 'Không có dữ liệu phân tích';
  document.getElementById('analysisBody').innerHTML = markdownToHtml(body);

  // Show result
  document.getElementById('resultArea').classList.add('active');

  // All steps done
  ['step1','step2','step3','step4'].forEach(s => {
    const el = document.getElementById(s);
    if(el){ el.classList.remove('active'); el.classList.add('done'); }
  });
}

// ── SIMPLE MARKDOWN RENDERER ──
function markdownToHtml(text) {
  if (!text) return '';
  return text
    // Remove DeepSeek <think> tags
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // Italic
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // H2
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    // H3
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    // H1
    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
    // HR
    .replace(/^---+$/gm, '<hr/>')
    // Bullet list
    .replace(/^[•\-\*] (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>(\n|$))+/g, m => `<ul>${m}</ul>`)
    // Numbered list
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    // Blockquote
    .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
    // Line breaks
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br/>')
    // Wrap in p
    .replace(/^(?!<[houlbp])(.+)$/gm, (m) => m.startsWith('<') ? m : `<p>${m}</p>`);
}

// ── COPY ANALYSIS ──
function copyAnalysis() {
  const text = document.getElementById('analysisBody').innerText;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector('.copy-btn');
    btn.textContent = '✅ Đã sao chép';
    setTimeout(() => btn.textContent = '📋 Sao chép', 2000);
  });
}
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════
# AGENT 1 — NEWS (DuckDuckGo, free)
# ═══════════════════════════════════════════
class NewsAgent:
    def __init__(self):
        try:
            from duckduckgo_search import DDGS
            self.available = True
        except Exception as e:
            logger.warning(f"DDG unavailable: {e}")
            self.available = False

    def _search(self, query, max_results=10):
        if not self.available:
            return []
        try:
            from duckduckgo_search import DDGS
            with DDGS() as d:
                return list(d.text(query, region="vn-vi", max_results=max_results))
        except Exception as e:
            logger.error(f"DDG error: {e}")
            return []

    def market_news(self, symbol):
        results = []
        for q in [
            f"{symbol} cổ phiếu kết quả kinh doanh 2025",
            f"{symbol} HOSE HNX tin tức mới nhất",
            f"VN-Index thị trường chứng khoán Việt Nam 2025",
        ]:
            results += self._search(q, max_results=5)
        seen = set()
        unique = []
        for r in results:
            t = r.get("title","")
            if t and t not in seen:
                seen.add(t)
                unique.append(r)
        return unique[:12]

    def fund_news(self, symbol):
        results = self._search(f"{symbol} chứng chỉ quỹ NAV hiệu suất 2025", max_results=8)
        results += self._search(f"quỹ mở ETF Việt Nam 2025", max_results=5)
        return results[:10]

    def forex_news(self, pair):
        results = self._search(f"tỷ giá {pair} NHNN chính sách tiền tệ 2025", max_results=8)
        results += self._search(f"{pair} forex technical analysis 2025", max_results=5)
        return results[:10]


# ═══════════════════════════════════════════
# AGENT 2 — DOCUMENT READER (Gemini, free)
# ═══════════════════════════════════════════
class DocumentAgent:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY_STOCK", os.getenv("GEMINI_API_KEY", ""))
        self.available = bool(api_key)
        self.client = None
        if self.available:
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
                self.model = "gemini-2.0-flash"
                logger.info("Gemini OK")
            except Exception as e:
                logger.error(f"Gemini init: {e}")
                self.available = False

    def extract_pdf(self, path):
        try:
            import pdfplumber
            txt = ""
            with pdfplumber.open(path) as pdf:
                for pg in pdf.pages:
                    t = pg.extract_text()
                    if t: txt += t + "\n"
            return txt[:60000]
        except Exception as e:
            logger.error(f"PDF error: {e}")
            return ""

    def analyze(self, text, symbol):
        if not self.available or not text:
            return {"summary": "", "available": False}
        try:
            prompt = f"""Bạn là CFA chuyên phân tích tài chính doanh nghiệp Việt Nam.
Phân tích tài liệu tài chính của **{symbol}**, trích xuất đầy đủ:

**A. CHỈ SỐ TÀI CHÍNH CHÍNH:**
- Doanh thu, EBITDA, LNST (3 năm gần nhất + tăng trưởng YoY)
- EPS, P/E, P/B, EV/EBITDA
- ROE, ROA, ROIC
- Tỷ lệ nợ/vốn chủ (D/E), Current Ratio, Quick Ratio
- Dòng tiền tự do (FCF)

**B. PHÂN TÍCH CHẤT LƯỢNG:**
- Chất lượng lợi nhuận (accruals, cash conversion)
- Hiệu quả sử dụng vốn
- Rủi ro tài chính (đòn bẩy, thanh khoản)

**C. SO SÁNH NGÀNH:**
- Định giá so với trung bình ngành
- Điểm mạnh/yếu cạnh tranh

Tài liệu:
{text[:25000]}

Trả lời tiếng Việt, súc tích, có số liệu cụ thể."""
            resp = self.client.models.generate_content(model=self.model, contents=prompt)
            return {"summary": resp.text, "available": True}
        except Exception as e:
            logger.error(f"Gemini analyze: {e}")
            return {"summary": f"Lỗi Gemini: {e}", "available": False}


# ═══════════════════════════════════════════
# AGENT 3 — REASONING (Groq/DeepSeek-R1, free)
# ═══════════════════════════════════════════
class ReasoningAgent:
    MODELS = ["deepseek-r1-distill-llama-70b", "llama-3.3-70b-versatile", "llama3-70b-8192"]

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY_STOCK", os.getenv("GROQ_API_KEY", ""))
        self.available = bool(api_key)
        self.client = None
        if self.available:
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
                logger.info("Groq OK")
            except Exception as e:
                logger.error(f"Groq init: {e}")
                self.available = False

    def _call(self, system, user, max_tokens=6000):
        if not self.available:
            return "⚠ Chưa cấu hình GROQ_API_KEY_STOCK. Vào Render → Environment → kiểm tra key."
        for model in self.MODELS:
            try:
                r = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role":"system","content":system},{"role":"user","content":user}],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                return r.choices[0].message.content
            except Exception as e:
                logger.warning(f"Groq {model}: {e}")
        return "❌ Không thể kết nối Groq API. Kiểm tra API key."

    # ── STOCK ──
    def analyze_stock(self, symbol, news, doc, stock_type="stock"):
        label = "chứng chỉ quỹ" if stock_type == "fund" else "cổ phiếu"
        news_txt = "\n".join([f"• {n.get('title','')} — {n.get('body','')[:300]}" for n in news[:8]])
        doc_txt  = doc.get("summary", "Không có tài liệu")
        doc_note = "✅ Đã phân tích tài liệu tài chính" if doc.get("available") else "⚠ Không có tài liệu PDF"

        system = """Bạn là chuyên gia phân tích chứng khoán cấp cao (CFA, CAIA) với 20+ năm kinh nghiệm tại thị trường Việt Nam.
Nhiệm vụ: Viết báo cáo phân tích đầu tư chuyên nghiệp, sâu sắc, có căn cứ số liệu.
Phong cách: Như báo cáo của SSI Research / VCSC / Maybank IB — rõ ràng, có luận điểm, không chung chung.
Quan trọng: Luôn đưa ra con số cụ thể, vùng giá cụ thể, tỷ lệ cụ thể."""

        user = f"""# YÊU CẦU PHÂN TÍCH CHUYÊN SÂU: {label.upper()} **{symbol}**

## NGUỒN DỮ LIỆU
**Tin tức thị trường ({len(news)} bài):**
{news_txt or "Không có tin tức"}

**Phân tích tài liệu:** {doc_note}
{doc_txt}

---
## CẤU TRÚC BÁO CÁO YÊU CẦU

### 📊 1. TÓM TẮT ĐIỀU HÀNH (Executive Summary)
- Luận điểm đầu tư chính (1 đoạn súc tích)
- Khuyến nghị: **[MUA / BÁN / GIỮ / THEO DÕI]** + Giá mục tiêu
- Risk/Reward ratio ước tính

### 🌍 2. BỐI CẢNH VĨ MÔ & NGÀNH
- Kinh tế Việt Nam: GDP, lạm phát, lãi suất, tỷ giá ảnh hưởng
- Chu kỳ ngành và vị thế của {symbol} trong ngành
- Xu hướng dài hạn tác động (ESG, chuyển đổi số, dịch chuyển chuỗi cung ứng...)

### 💰 3. PHÂN TÍCH CƠ BẢN CHUYÊN SÂU
#### 3.1 Sức khỏe tài chính
- Doanh thu & tăng trưởng (ước tính nếu không có số liệu)
- Biên lợi nhuận (gross/EBIT/net margin)
- Dòng tiền & chất lượng lợi nhuận
- Cấu trúc nợ & khả năng trả nợ

#### 3.2 Định giá (Valuation)
- P/E forward ước tính vs ngành vs lịch sử
- P/B so với ROE → xác định rẻ/đắt
- Phương pháp DCF: ước tính fair value
- So sánh với peers trong ngành

#### 3.3 Lợi thế cạnh tranh (Moat)
- Lợi thế bền vững là gì?
- Rủi ro mất thị phần

### 📈 4. PHÂN TÍCH KỸ THUẬT
#### 4.1 Xu hướng & Momentum
- Xu hướng ngắn hạn (1-4 tuần): Tăng/Giảm/Đi ngang
- Xu hướng trung hạn (1-3 tháng)
- Xu hướng dài hạn (6-12 tháng)

#### 4.2 Các mức giá quan trọng
- **Kháng cự**: Mức 1, Mức 2, Mức 3
- **Hỗ trợ**: Mức 1, Mức 2, Mức 3

#### 4.3 Chỉ báo kỹ thuật
- RSI(14): Vùng hiện tại → quá mua/bán/trung lập
- MACD: Tín hiệu bullish/bearish
- MA20/MA50/MA200: Vị trí giá so với các đường MA
- Bollinger Bands: Vùng nén/giãn
- Volume: Xác nhận xu hướng

#### 4.4 Mô hình & Pattern
- Mô hình giá đáng chú ý (nếu có)
- Fibonacci retracement levels

### 💵 5. PHÂN TÍCH GIÁ & ĐỊNH GIÁ TỔNG HỢP
- **Fair Value** (giá trị hợp lý): ước tính cụ thể
- **Bull case** (kịch bản lạc quan): giá mục tiêu + điều kiện
- **Base case** (kịch bản cơ sở): giá mục tiêu + xác suất
- **Bear case** (kịch bản bi quan): downside + điều kiện
- Upside/Downside tiềm năng (%)

### ✅ 6. KHUYẾN NGHỊ ĐẦU TƯ CHI TIẾT
**QUYẾT ĐỊNH: [MUA / BÁN / GIỮ / THEO DÕI]**

| Thông số | Giá trị |
|----------|---------|
| Giá mục tiêu 3 tháng | xxx |
| Giá mục tiêu 6 tháng | xxx |
| Điểm vào lệnh lý tưởng | xxx |
| Stop-loss | xxx |
| Tỷ lệ Risk/Reward | x:x |
| % danh mục đề xuất | x% |
| Thời hạn nắm giữ | xxx |

- **Luận điểm MUA/BÁN/GIỮ** (3-5 điểm cụ thể)
- **Điều kiện để thay đổi khuyến nghị**

### ⚠️ 7. RỦI RO & CATALYST
#### Rủi ro chính (xếp hạng theo mức độ)
1. Rủi ro #1 + xác suất + tác động
2. Rủi ro #2
3. Rủi ro #3

#### Catalyst tích cực sắp tới
- Sự kiện 1 + thời gian dự kiến
- Sự kiện 2

### 📅 8. LỊCH SỰ KIỆN CẦN THEO DÕI
- ĐHCĐ, công bố KQKD, ngày GDRCC, sự kiện ngành...

---
Viết đầy đủ, chuyên nghiệp, bằng tiếng Việt. Dùng số liệu ước tính có căn cứ nếu không có data cụ thể. Dùng emoji phù hợp."""

        result = self._call(system, user, max_tokens=6000)
        return {
            "analysis": result,
            "recommendation": self._extract_rec(result),
            "symbol": symbol,
            "type": stock_type,
        }

    # ── FOREX ──
    def analyze_forex(self, pair, news):
        news_txt = "\n".join([f"• {n.get('title','')} — {n.get('body','')[:300]}" for n in news[:8]])
        parts = pair.upper().replace(".", "/")

        system = """Bạn là chuyên gia phân tích ngoại hối (FX) cấp cao với kinh nghiệm tại các NHTM lớn Việt Nam.
Phân tích sâu, có số liệu cụ thể, phong cách như báo cáo FX của Vietcombank/BIDV Research."""

        user = f"""# PHÂN TÍCH TỶ GIÁ CHUYÊN SÂU: **{parts}**

## TIN TỨC & DỮ LIỆU
{news_txt or "Không có tin tức"}

---
## YÊU CẦU BÁO CÁO

### 🌐 1. TỔNG QUAN CẶP TIỀN {parts}
- Đặc điểm & thanh khoản của cặp tiền
- Tầm quan trọng với kinh tế Việt Nam (nếu liên quan VND)
- Bối cảnh địa kinh tế hiện tại

### 📊 2. PHÂN TÍCH CƠ BẢN (Fundamental)
#### Yếu tố vĩ mô
- Chính sách tiền tệ: Fed/ECB/BOJ/NHNN → hướng lãi suất
- Chênh lệch lãi suất (interest rate differential)
- Lạm phát 2 nước: CPI gần nhất, dự báo
- Tăng trưởng kinh tế: GDP, PMI, việc làm
- Cán cân thương mại & dòng vốn FDI/FII

#### Yếu tố địa chính trị & thị trường
- Rủi ro địa chính trị ảnh hưởng
- Tâm lý thị trường (risk-on/risk-off)
- Vị thế của USD Index (DXY) nếu liên quan

### 📈 3. PHÂN TÍCH KỸ THUẬT
#### Xu hướng
- Ngắn hạn (1-2 tuần): 
- Trung hạn (1-3 tháng):
- Dài hạn (6-12 tháng):

#### Mức giá quan trọng
- **Kháng cự**: R1, R2, R3 (giá cụ thể)
- **Hỗ trợ**: S1, S2, S3 (giá cụ thể)
- Pivot point ngày/tuần

#### Chỉ báo kỹ thuật
- RSI(14), MACD, Stochastic
- MA50/MA200 & Golden/Death Cross
- Bollinger Bands
- ATR (độ biến động trung bình)

### 🔮 4. DỰ BÁO TỶ GIÁ
| Kỳ hạn | Dự báo | Kịch bản lạc quan | Kịch bản bi quan |
|--------|--------|-------------------|-----------------|
| 1 tuần | | | |
| 1 tháng | | | |
| 3 tháng | | | |

### 💼 5. TƯ VẤN CHIẾN LƯỢC
**NHẬN ĐỊNH: [TỶ GIÁ DỰ KIẾN TĂNG / GIẢM / ĐI NGANG]**

#### Doanh nghiệp xuất khẩu
- Chiến lược bán ngoại tệ kỳ hạn
- Mức tỷ giá nên chốt

#### Doanh nghiệp nhập khẩu
- Chiến lược mua ngoại tệ
- Timing phù hợp

#### Nhà đầu tư cá nhân
- Nên mua/bán/giữ ngoại tệ?
- Mức giá target & stop-loss

### ⚠️ 6. RỦI RO & SỰ KIỆN CẦN THEO DÕI
- Top 3 rủi ro tăng/giảm tỷ giá
- Lịch sự kiện kinh tế quan trọng (FOMC, CPI, NFP...)

Viết đầy đủ, chuyên nghiệp, tiếng Việt, có số liệu ước tính cụ thể."""

        result = self._call(system, user, max_tokens=5000)
        return {
            "analysis": result,
            "direction": self._extract_dir(result),
            "pair": pair,
        }

    def _extract_rec(self, t):
        u = t.upper()
        for kw, val in [
            (["**MUA**","[MUA]","KHUYẾN NGHỊ: MUA","KHUYẾN NGHỊ MUA","QUYẾT ĐỊNH: MUA"], "BUY"),
            (["**BÁN**","[BÁN]","KHUYẾN NGHỊ: BÁN","KHUYẾN NGHỊ BÁN","QUYẾT ĐỊNH: BÁN"], "SELL"),
            (["**GIỮ**","[GIỮ]","KHUYẾN NGHỊ: GIỮ","QUYẾT ĐỊNH: GIỮ"], "HOLD"),
        ]:
            if any(k in u for k in kw): return val
        return "WATCH"

    def _extract_dir(self, t):
        u = t.upper()
        if any(k in u for k in ["DỰ KIẾN TĂNG","[TĂNG]","TỶ GIÁ TĂNG","NHẬN ĐỊNH: TĂNG"]): return "UP"
        if any(k in u for k in ["DỰ KIẾN GIẢM","[GIẢM]","TỶ GIÁ GIẢM","NHẬN ĐỊNH: GIẢM"]): return "DOWN"
        return "SIDEWAYS"


# ═══════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════
class Orchestrator:
    def __init__(self):
        self.news = NewsAgent()
        self.doc  = DocumentAgent()
        self.ai   = ReasoningAgent()

    def run_stock(self, symbol, pdf_paths=None, stock_type="stock"):
        if stock_type == "fund":
            news = self.news.fund_news(symbol)
        else:
            news = self.news.market_news(symbol)
        doc = {"summary": "", "available": False}
        if pdf_paths:
            txt = ""
            for p in pdf_paths:
                txt += self.doc.extract_pdf(p) + "\n\n"
                try: os.remove(p)
                except: pass
            if txt.strip():
                doc = self.doc.analyze(txt, symbol)
        r = self.ai.analyze_stock(symbol, news, doc, stock_type)
        r["news_count"]    = len(news)
        r["has_documents"] = bool(pdf_paths)
        return r

    def run_forex(self, pair):
        news = self.news.forex_news(pair)
        r = self.ai.analyze_forex(pair, news)
        r["news_count"] = len(news)
        return r


orc = Orchestrator()


# ═══════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════
@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "agents": {
            "news":      orc.news.available,
            "document":  orc.doc.available,
            "reasoning": orc.ai.available,
        }
    })

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        symbol = request.form.get("symbol", "").strip().upper()
        atype  = request.form.get("type", "stock")
        if not symbol:
            return jsonify({"error": "Vui lòng nhập mã"}), 400
        # Forex detection
        if atype == "forex" or ("." in symbol and len(symbol) <= 8):
            return jsonify({"success": True, "data": orc.run_forex(symbol), "mode": "forex"})
        # PDF uploads
        pdf_paths = []
        if "pdfs" in request.files:
            for f in request.files.getlist("pdfs"):
                if f and f.filename.lower().endswith(".pdf"):
                    p = str(TEMP_DIR / f"{uuid.uuid4()}.pdf")
                    f.save(p)
                    pdf_paths.append(p)
        return jsonify({"success": True, "data": orc.run_stock(symbol, pdf_paths, atype), "mode": "stock"})
    except Exception as e:
        logger.error(f"/api/analyze error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/scrape-url", methods=["POST"])
def scrape_url():
    try:
        url = (request.get_json() or {}).get("url", "")
        if not url:
            return jsonify({"error": "No URL"}), 400
        import requests as req
        from bs4 import BeautifulSoup
        resp = req.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")
        for t in soup(["script","style","nav","footer"]): t.decompose()
        return jsonify({"success": True, "content": soup.get_text(separator="\n", strip=True)[:5000]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
