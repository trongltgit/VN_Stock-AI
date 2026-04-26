// ── State ──────────────────────────────────────────────────────────────────
let state = {
  assetType: 'stock',
  history: [],
  sources: [
    { name: 'VCBS Quotes',   url: 'https://quotes.vcbs.com.vn/k8/',       type: 'broker',   active: true  },
    { name: 'VCBF Fund',     url: 'https://www.vcbf.com/',                 type: 'fund',     active: true  },
    { name: 'CafeF',         url: 'https://cafef.vn',                      type: 'news',     active: true  },
    { name: 'VietStock',     url: 'https://finance.vietstock.vn',          type: 'data',     active: true  },
    { name: 'FireAnt',       url: 'https://fireant.vn',                    type: 'data',     active: true  },
    { name: 'SSI Research',  url: 'https://www.ssi.com.vn',                type: 'research', active: false },
    { name: 'DNSE',          url: 'https://www.dnse.com.vn',               type: 'broker',   active: false },
    { name: 'Vndirect',      url: 'https://www.vndirect.com.vn',           type: 'broker',   active: false },
  ]
};

// ── Init ───────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  renderSources();
});

async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    const dot = document.querySelector('.status-dot');
    const txt = document.querySelector('.status-text');
    if (r.ok) {
      dot.classList.add('ok');
      txt.textContent = 'API sẵn sàng';
    } else {
      dot.classList.add('err');
      txt.textContent = 'Lỗi kết nối';
    }
  } catch {
    document.querySelector('.status-dot').classList.add('err');
    document.querySelector('.status-text').textContent = 'Lỗi kết nối';
  }
}

// ── Tab navigation ─────────────────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
  document.getElementById(`panel-${tab}`).classList.add('active');
  if (tab === 'sources') renderSources();
  if (tab === 'history') renderHistory();
}

// ── Asset type ─────────────────────────────────────────────────────────────
function setAsset(el, type) {
  state.assetType = type;
  document.querySelectorAll('#asset-chips .chip').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  const ph = { stock: 'VD: VCB, TCB, MBB, VNM, HPG, VCBS...', fund: 'VD: VCBF-BCF, VCBF-FIF, MGF, VFB...', etf: 'VD: E1VFVN30, FUEVFVND, FUESSVFL...', bond: 'VD: VCB11B2227, TCB11B2226...' };
  document.getElementById('ticker-input').placeholder = ph[type] || 'Nhập mã...';
}

function toggleChip(el) { el.classList.toggle('active'); }

function quickPick(ticker) {
  document.getElementById('ticker-input').value = ticker;
  document.getElementById('ticker-input').focus();
}

// ── Analyze ────────────────────────────────────────────────────────────────
async function runAnalysis() {
  const ticker = document.getElementById('ticker-input').value.trim().toUpperCase();
  if (!ticker) { showError('result-area', 'Vui lòng nhập mã chứng khoán.'); return; }

  const analysisTypes = [...document.querySelectorAll('#analysis-chips .chip.active')]
    .map(c => c.textContent.trim());
  const sources = state.sources.filter(s => s.active).map(s => s.name);

  const btn = document.getElementById('analyze-btn');
  btn.disabled = true;

  showLoading('result-area', ticker, [
    'Thu thập dữ liệu từ các nguồn...',
    'Phân tích cơ bản & tài chính...',
    'Phân tích kỹ thuật & đồ thị...',
    'Phân tích xu hướng giá...',
    'Tổng hợp & tạo khuyến nghị AI...'
  ]);

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker, assetType: state.assetType, analysisTypes, sources })
    });
    const json = await res.json();
    btn.disabled = false;

    if (!res.ok || json.error) { showError('result-area', json.error || 'Lỗi không xác định.'); return; }
    if (json.raw) { showRaw('result-area', json.raw); return; }

    renderResult(json.data, ticker);
    state.history.unshift({ ticker, verdict: json.data.verdict, company: json.data.company, time: new Date().toLocaleTimeString('vi-VN'), data: json.data });
    if (state.history.length > 30) state.history.pop();
  } catch (err) {
    btn.disabled = false;
    showError('result-area', 'Không thể kết nối đến server. Vui lòng thử lại.');
  }
}

// ── Market ─────────────────────────────────────────────────────────────────
async function loadMarket() {
  const btn = document.getElementById('market-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner" style="width:14px;height:14px;border-width:2px;display:inline-block;margin-right:6px;"></span> Đang phân tích...';

  try {
    const res = await fetch('/api/market', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
    const json = await res.json();
    btn.disabled = false;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 15 15" fill="none"><path d="M8 1.5A6.5 6.5 0 1 1 1.5 8" stroke="white" stroke-width="1.5" stroke-linecap="round"/><polyline points="1,5 1.5,8 4.5,7.5" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg> Làm mới phân tích';

    if (!res.ok || json.error) { document.getElementById('market-result').innerHTML = `<div class="empty-state">${json.error || 'Lỗi phân tích.'}</div>`; return; }

    const d = json.data;

    // Update index cards
    const updateIdx = (prefix, obj) => {
      const v = document.getElementById(`${prefix}-val`);
      const c = document.getElementById(`${prefix}-chg`);
      if (v) v.textContent = obj.value || '—';
      if (c) { c.textContent = `${obj.change || ''} (${obj.change_pct || ''})`; c.className = 'idx-chg ' + (obj.trend === 'Tăng' ? 'up' : obj.trend === 'Giảm' ? 'down' : 'neutral'); }
    };
    if (d.vnindex) updateIdx('vnindex', d.vnindex);
    if (d.hnx) updateIdx('hnx', d.hnx);
    if (d.upcom) updateIdx('upcom', d.upcom);

    const sentColor = d.market_sentiment === 'Tích cực' ? 'up' : d.market_sentiment === 'Tiêu cực' ? 'down' : 'neutral';
    const hotSectors = (d.hot_sectors || []).map(s => `<span class="tag">${s}</span>`).join('');
    const topPicks = (d.top_picks || []).map(t => `<span class="tag">${t}</span>`).join('');
    const avoidList = (d.avoid_list || []).map(t => `<span class="tag tag-warn">${t}</span>`).join('');

    document.getElementById('market-result').innerHTML = `
      <div class="result-card">
        <div class="result-header">
          <div>
            <div class="result-ticker">Tổng quan thị trường</div>
            <div class="result-company">${d.market_phase || ''}</div>
          </div>
          <div class="verdict-wrap">
            <span class="verdict-badge ${sentColor === 'up' ? 'verdict-buy' : sentColor === 'down' ? 'verdict-sell' : 'verdict-hold'}">${d.market_sentiment || '—'}</span>
          </div>
        </div>
        <div class="analysis-body">
          <div class="analysis-block blue">
            <h3>📊 Phân tích vĩ mô</h3>
            <p>${d.macro_factors || '—'}</p>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
            <div class="analysis-block">
              <h3>📈 Nhận định 1 tháng</h3>
              <p>${d.outlook_1m || '—'}</p>
            </div>
            <div class="analysis-block">
              <h3>📅 Nhận định 3 tháng</h3>
              <p>${d.outlook_3m || '—'}</p>
            </div>
          </div>
          <div class="analysis-block green">
            <h3>💡 Chiến lược được khuyến nghị</h3>
            <p>${d.strategy || '—'}</p>
          </div>
          ${hotSectors ? `<div><div class="form-label" style="margin-bottom:8px;">Ngành nóng</div><div class="tag-cloud">${hotSectors}</div></div>` : ''}
          ${topPicks ? `<div><div class="form-label" style="margin-bottom:8px;">Mã đáng chú ý</div><div class="tag-cloud">${topPicks}</div></div>` : ''}
          ${avoidList ? `<div><div class="form-label" style="margin-bottom:8px;">Cần cẩn trọng</div><div class="tag-cloud">${avoidList}</div></div>` : ''}
          ${d.note ? `<p style="font-size:12px;color:var(--text-3);font-style:italic;">⚠️ ${d.note}</p>` : ''}
        </div>
      </div>`;
  } catch {
    btn.disabled = false;
    document.getElementById('market-result').innerHTML = '<div class="empty-state">Lỗi kết nối server.</div>';
  }
}

// ── Compare ────────────────────────────────────────────────────────────────
async function runCompare() {
  const raw = document.getElementById('compare-input').value.trim().toUpperCase();
  if (!raw) return;
  const tickers = raw.split(',').map(t => t.trim()).filter(Boolean).slice(0, 4);
  if (tickers.length < 2) { showError('compare-result', 'Nhập ít nhất 2 mã để so sánh.'); return; }

  showLoading('compare-result', tickers.join(' & '), ['Đang phân tích từng mã...', 'Đang tổng hợp so sánh...']);

  try {
    const results = await Promise.all(tickers.map(ticker =>
      fetch('/api/analyze', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker, assetType: 'stock', analysisTypes: ['Phân tích cơ bản', 'Phân tích kỹ thuật', 'Phân tích giá'], sources: state.sources.filter(s => s.active).map(s => s.name) })
      }).then(r => r.json())
    ));

    const cards = results.map((r, i) => {
      if (!r.success || !r.data) return `<div class="result-card" style="padding:1rem;"><b>${tickers[i]}</b>: Không thể phân tích.</div>`;
      const d = r.data;
      const vc = d.verdict === 'MUA' ? 'verdict-buy' : d.verdict === 'BÁN' ? 'verdict-sell' : 'verdict-hold';
      return `
        <div class="result-card">
          <div class="result-header">
            <div><div class="result-ticker">${tickers[i]}</div><div class="result-company">${d.company || ''}</div></div>
            <span class="verdict-badge ${vc}">${d.verdict}</span>
          </div>
          <div class="metrics-grid" style="grid-template-columns:repeat(2,1fr);">
            <div class="metric-cell"><div class="metric-label">Giá</div><div class="metric-value" style="font-size:16px;">${d.price || '—'}</div></div>
            <div class="metric-cell"><div class="metric-label">P/E</div><div class="metric-value" style="font-size:16px;">${d.pe_ratio || '—'}x</div></div>
          </div>
          <div style="padding:1rem 1.25rem;">
            <div class="scores-grid" style="grid-template-columns:repeat(2,1fr);margin-bottom:10px;">
              ${[['Cơ bản', d.fundamental_score], ['Kỹ thuật', d.technical_score]].map(([l, s]) => scoreItem(l, s)).join('')}
            </div>
            <p style="font-size:13px;color:var(--text-2);line-height:1.6;">${d.ai_recommendation ? d.ai_recommendation.substring(0, 200) + '...' : ''}</p>
          </div>
        </div>`;
    });

    document.getElementById('compare-result').innerHTML = `<div style="display:grid;grid-template-columns:repeat(${Math.min(tickers.length, 2)},1fr);gap:12px;margin-top:1rem;">${cards.join('')}</div>`;
  } catch {
    showError('compare-result', 'Lỗi so sánh. Vui lòng thử lại.');
  }
}

// ── Render result ──────────────────────────────────────────────────────────
function renderResult(d, ticker) {
  const vc = d.verdict === 'MUA' ? 'verdict-buy' : d.verdict === 'BÁN' ? 'verdict-sell' : 'verdict-hold';
  const vb = d.verdict === 'MUA' ? 'buy' : d.verdict === 'BÁN' ? 'sell' : 'hold';
  const sc = d.change_pct && d.change_pct.startsWith('+') ? 'up' : 'down';

  const signals = (d.signals || []).map(s => {
    const cls = s.type === 'positive' ? 'signal-pos' : s.type === 'negative' ? 'signal-neg' : 'signal-neu';
    const dot = s.type === 'positive' ? 'sdot-pos' : s.type === 'negative' ? 'sdot-neg' : 'sdot-neu';
    return `<div class="signal-item ${cls}"><div class="signal-dot ${dot}"></div>${s.label}</div>`;
  }).join('');

  const catalysts = (d.catalysts || []).map(c => `<li>${c}</li>`).join('');
  const risks = (d.risks || []).map(r => `<li>${r}</li>`).join('');

  document.getElementById('result-area').innerHTML = `
    <div class="result-card">
      <div class="result-header">
        <div class="result-ticker-block">
          <div class="result-ticker">${ticker}</div>
          <div class="result-company">${d.company || ''} · ${d.sector || ''}</div>
          <span class="result-exchange">${d.exchange || ''}</span>
        </div>
        <div class="verdict-wrap">
          <div class="verdict-badge ${vc}">${d.verdict || '—'}</div>
          <div class="verdict-sub">Độ tin cậy: ${d.verdict_confidence || 0}%</div>
          <div class="verdict-sub">Rủi ro: ${d.risk_level || '—'}</div>
        </div>
      </div>

      <div class="metrics-grid">
        <div class="metric-cell">
          <div class="metric-label">Giá tham chiếu</div>
          <div class="metric-value">${d.price || '—'}</div>
          <div class="metric-sub ${sc}">${d.change_pct || ''}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-label">P/E</div>
          <div class="metric-value">${d.pe_ratio || 'N/A'}x</div>
          <div class="metric-sub neutral">EPS: ${d.eps || 'N/A'}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-label">P/B</div>
          <div class="metric-value">${d.pb_ratio || 'N/A'}x</div>
          <div class="metric-sub neutral">ROE: ${d.roe || 'N/A'}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-label">Vốn hóa</div>
          <div class="metric-value" style="font-size:15px;">${d.market_cap || '—'}</div>
          <div class="metric-sub neutral">52W: ${d.week52_low || '?'} – ${d.week52_high || '?'}</div>
        </div>
      </div>

      <div class="analysis-body">
        <!-- Scores -->
        <div class="scores-grid">
          ${scoreItem('Cơ bản', d.fundamental_score)}
          ${scoreItem('Kỹ thuật', d.technical_score)}
          ${scoreItem('Giá', d.price_score)}
          ${scoreItem('Thị trường', d.market_score)}
        </div>

        <!-- Price targets -->
        <div>
          <div class="form-label" style="margin-bottom:8px;">Mục tiêu giá & Cắt lỗ</div>
          <div class="price-targets">
            <div class="price-target">
              <div class="pt-label">Mục tiêu 3 tháng</div>
              <div class="pt-value pt-up">${d.target_price_3m || '—'}</div>
            </div>
            <div class="price-target">
              <div class="pt-label">Mục tiêu 1 năm</div>
              <div class="pt-value pt-up">${d.target_price_1y || '—'}</div>
            </div>
            <div class="price-target">
              <div class="pt-label">Cắt lỗ</div>
              <div class="pt-value pt-down">${d.stop_loss || '—'}</div>
            </div>
          </div>
        </div>

        <!-- Analysis blocks -->
        <div class="analysis-block blue">
          <h3>📊 Phân tích cơ bản</h3>
          <p>${d.fundamental_summary || '—'}</p>
        </div>
        <div class="analysis-block">
          <h3>📈 Phân tích kỹ thuật</h3>
          <p>${d.technical_summary || '—'}</p>
        </div>
        <div class="analysis-block">
          <h3>🕯️ Phân tích đồ thị</h3>
          <p>${d.chart_summary || '—'}</p>
        </div>
        <div class="analysis-block amber">
          <h3>💰 Phân tích giá & vùng giao dịch</h3>
          <p>${d.price_summary || '—'}</p>
        </div>
        <div class="analysis-block">
          <h3>🌐 Tổng quan thị trường</h3>
          <p>${d.market_summary || '—'}</p>
        </div>

        <!-- Verdict -->
        <div class="verdict-block ${vb}">
          <h3>⭐ Khuyến nghị AI — ${d.verdict || '—'}</h3>
          <p>${d.ai_recommendation || '—'}</p>
        </div>

        <!-- Catalysts & Risks -->
        ${(catalysts || risks) ? `
        <div class="lists-row">
          ${catalysts ? `<div class="mini-list catalyst"><h4>Động lực tăng</h4><ul>${catalysts}</ul></div>` : ''}
          ${risks ? `<div class="mini-list risk-list"><h4>Rủi ro cần chú ý</h4><ul>${risks}</ul></div>` : ''}
        </div>` : ''}

        <!-- Signals -->
        ${signals ? `<div><div class="form-label" style="margin-bottom:8px;">Tín hiệu giao dịch</div><div class="signals-grid">${signals}</div></div>` : ''}

        <!-- Price note -->
        ${d.price_note ? `<p style="font-size:12px;color:var(--text-3);font-style:italic;">⚠️ ${d.price_note}</p>` : ''}

        <!-- Action buttons -->
        <div class="action-bar">
          <button class="btn-secondary" onclick="sendCompare('${ticker}')">So sánh với ngành</button>
          <button class="btn-secondary" onclick="sendRisk('${ticker}')">Phân tích rủi ro sâu</button>
          <button class="btn-secondary" onclick="sendOutlook('${ticker}')">Triển vọng 2025–2026</button>
        </div>
      </div>
    </div>`;
}

function scoreItem(label, score) {
  const s = parseInt(score) || 0;
  const color = s >= 70 ? 'var(--green-600)' : s >= 50 ? 'var(--amber-600)' : 'var(--red-600)';
  return `<div class="score-item">
    <div class="score-header">
      <span class="score-label">${label}</span>
      <span class="score-num" style="color:${color}">${s}/100</span>
    </div>
    <div class="score-bar"><div class="score-fill" style="width:${s}%;background:${color};"></div></div>
  </div>`;
}

// ── Sources ────────────────────────────────────────────────────────────────
function renderSources() {
  document.getElementById('sources-list').innerHTML = state.sources.map((s, i) => `
    <div class="source-item">
      <div class="source-info">
        <div class="source-dot" style="background:${s.active ? 'var(--green-400)' : 'var(--gray-200)'};"></div>
        <div>
          <div class="source-name">${s.name}</div>
          <div class="source-url">${s.url}</div>
        </div>
      </div>
      <div class="source-actions">
        <span class="source-badge">${s.type}</span>
        <button class="btn-secondary" style="padding:5px 12px;font-size:12px;" onclick="toggleSource(${i})">${s.active ? 'Tắt' : 'Bật'}</button>
        <button class="btn-secondary" style="padding:5px 10px;font-size:12px;color:var(--red-600);" onclick="removeSource(${i})">✕</button>
      </div>
    </div>`).join('');
}

function toggleSource(i) { state.sources[i].active = !state.sources[i].active; renderSources(); }
function removeSource(i) { state.sources.splice(i, 1); renderSources(); }

function addSource() {
  const name = document.getElementById('source-name').value.trim();
  const url  = document.getElementById('source-url').value.trim();
  const type = document.getElementById('source-type').value;
  if (!url) return;
  const n = name || url.replace(/https?:\/\/(www\.)?/, '').split('/')[0];
  state.sources.push({ name: n, url, type, active: true });
  document.getElementById('source-name').value = '';
  document.getElementById('source-url').value  = '';
  renderSources();
}

// ── History ────────────────────────────────────────────────────────────────
function renderHistory() {
  const list = document.getElementById('history-list');
  if (!state.history.length) { list.innerHTML = '<div class="empty-state">Chưa có lịch sử phân tích trong phiên này.</div>'; return; }
  const vc = v => v === 'MUA' ? 'up' : v === 'BÁN' ? 'down' : 'neutral';
  list.innerHTML = state.history.map(h => `
    <div class="history-item" onclick="reloadFromHistory('${h.ticker}')">
      <div class="history-left">
        <div class="h-ticker">${h.ticker}</div>
        <div class="h-company">${h.company || ''}</div>
      </div>
      <div class="history-right">
        <span class="${vc(h.verdict)}" style="font-weight:600;font-size:13px;">${h.verdict}</span>
        <span class="h-time">${h.time}</span>
      </div>
    </div>`).join('');
}

function reloadFromHistory(ticker) {
  switchTab('analyze');
  const item = state.history.find(h => h.ticker === ticker);
  if (item && item.data) {
    document.getElementById('ticker-input').value = ticker;
    renderResult(item.data, ticker);
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────
function showLoading(targetId, label, steps) {
  const stepsHtml = steps.map((s, i) =>
    `<div class="lst ${i === 0 ? 'lst-active' : 'lst-wait'}" id="lstep-${i}">
      ${i === 0 ? '⟳' : '○'} ${s}
    </div>`).join('');
  document.getElementById(targetId).innerHTML = `
    <div class="result-card">
      <div class="loading-box">
        <div class="spinner"></div>
        <div class="loading-title">AI đang phân tích ${label}...</div>
        <div class="loading-steps">${stepsHtml}</div>
      </div>
    </div>`;
  let idx = 1;
  const t = setInterval(() => {
    if (idx >= steps.length) { clearInterval(t); return; }
    const prev = document.getElementById(`lstep-${idx - 1}`);
    if (prev) { prev.className = 'lst lst-done'; prev.textContent = '✓ ' + steps[idx - 1]; }
    const cur = document.getElementById(`lstep-${idx}`);
    if (cur) { cur.className = 'lst lst-active'; cur.textContent = '⟳ ' + steps[idx]; }
    idx++;
  }, 800);
}

function showError(targetId, msg) {
  document.getElementById(targetId).innerHTML = `
    <div class="result-card" style="padding:1.5rem;">
      <p style="color:var(--red-600);font-size:14px;">⚠️ ${msg}</p>
    </div>`;
}

function showRaw(targetId, raw) {
  document.getElementById(targetId).innerHTML = `
    <div class="result-card" style="padding:1.5rem;">
      <pre style="font-size:12px;color:var(--text-2);white-space:pre-wrap;line-height:1.6;">${raw}</pre>
    </div>`;
}

function sendCompare(ticker) {
  switchTab('compare');
  document.getElementById('compare-input').value = ticker + ', ';
  document.getElementById('compare-input').focus();
}

function sendRisk(ticker) {
  alert(`Để phân tích rủi ro sâu hơn, hãy hỏi trực tiếp AI của bạn:\n"Phân tích chi tiết rủi ro đầu tư vào ${ticker} và kịch bản giá trong 3 tháng tới"`);
}

function sendOutlook(ticker) {
  alert(`Để phân tích triển vọng, hãy hỏi AI:\n"Phân tích triển vọng tăng trưởng của ${ticker} năm 2025-2026 dựa trên báo cáo tài chính mới nhất"`);
}
