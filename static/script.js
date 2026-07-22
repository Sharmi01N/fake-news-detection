const input = document.getElementById('articleInput');
const urlInput = document.getElementById('urlInput');
const charCount = document.getElementById('charCount');
const analyzeBtn = document.getElementById('analyzeBtn');
const placeholder = document.getElementById('verdictPlaceholder');
const stampWrap = document.getElementById('stampWrap');
const stamp = document.getElementById('verdictStamp');
const stampWord = document.getElementById('stampWord');
const confidenceValue = document.getElementById('confidenceValue');
const verdictTitleRow = document.getElementById('verdictTitleRow');
const verdictTitle = document.getElementById('verdictTitle');
const errorBox = document.getElementById('verdictError');
const clock = document.getElementById('liveClock');

const modeTextBtn = document.getElementById('modeTextBtn');
const modeUrlBtn = document.getElementById('modeUrlBtn');

const wireList = document.getElementById('wireList');
const wireUpdated = document.getElementById('wireUpdated');
const wireEmpty = document.getElementById('wireEmpty');

let mode = 'text'; // 'text' | 'url'

function updateClock() {
  const now = new Date();
  clock.textContent = now.toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' });
}
updateClock();
setInterval(updateClock, 60000);

/* ---------- Mode toggle ---------- */

function setMode(next) {
  mode = next;
  const isText = mode === 'text';

  modeTextBtn.classList.toggle('is-active', isText);
  modeTextBtn.setAttribute('aria-selected', isText);
  modeUrlBtn.classList.toggle('is-active', !isText);
  modeUrlBtn.setAttribute('aria-selected', !isText);

  input.hidden = !isText;
  urlInput.hidden = isText;
}

modeTextBtn.addEventListener('click', () => setMode('text'));
modeUrlBtn.addEventListener('click', () => setMode('url'));

input.addEventListener('input', () => {
  charCount.textContent = `${input.value.length} characters`;
});
urlInput.addEventListener('input', () => {
  charCount.textContent = mode === 'url' ? 'URL will be fetched live' : `${input.value.length} characters`;
});

/* ---------- Verdict stage ---------- */

function resetStage() {
  stampWrap.hidden = true;
  errorBox.hidden = true;
  placeholder.hidden = false;
  stamp.classList.remove('is-shown', 'is-real');
  verdictTitleRow.hidden = true;
}

function showError(message) {
  placeholder.hidden = true;
  stampWrap.hidden = true;
  errorBox.hidden = false;
  errorBox.textContent = message;
}

function showVerdict(data) {
  placeholder.hidden = true;
  errorBox.hidden = true;
  stampWrap.hidden = false;

  stamp.classList.remove('is-real');
  if (data.label === 'real') stamp.classList.add('is-real');
  stampWord.textContent = data.label === 'fake' ? 'FAKE' : 'REAL';

  confidenceValue.textContent = data.confidence != null ? `${data.confidence}%` : 'n/a';

  if (data.title) {
    verdictTitleRow.hidden = false;
    verdictTitle.textContent = data.title;
  } else {
    verdictTitleRow.hidden = true;
  }

  // restart the stamp animation
  stamp.classList.remove('is-shown');
  void stamp.offsetWidth;
  requestAnimationFrame(() => stamp.classList.add('is-shown'));
}

async function analyze() {
  const payload = mode === 'url'
    ? { url: urlInput.value.trim() }
    : { text: input.value.trim() };

  if (mode === 'url' && !payload.url) {
    showError('Paste a news article URL above before sending it to the desk.');
    return;
  }
  if (mode === 'text' && !payload.text) {
    showError('Paste some text above before sending it to the desk.');
    return;
  }

  analyzeBtn.classList.add('is-loading');
  analyzeBtn.disabled = true;

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'Something went wrong reviewing this clipping.');
      return;
    }
    showVerdict(data);
  } catch (err) {
    showError('Could not reach the desk. Is the Flask server running?');
  } finally {
    analyzeBtn.classList.remove('is-loading');
    analyzeBtn.disabled = false;
  }
}

analyzeBtn.addEventListener('click', analyze);
input.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') analyze();
});
urlInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') analyze();
});

resetStage();

/* ---------- Live Wire ---------- */

function timeAgo(seconds) {
  if (!seconds || seconds <= 0) return 'just now';
  if (seconds < 60) return `${Math.floor(seconds)}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

function renderWire(items, updatedTs) {
  wireList.innerHTML = '';

  if (!items || items.length === 0) {
    wireList.appendChild(wireEmpty);
    wireEmpty.textContent = 'No live headlines available right now.';
    return;
  }

  for (const item of items) {
    const li = document.createElement('li');
    li.className = 'wire__item';

    const badgeClass = item.label === 'fake'
      ? 'wire__badge wire__badge--fake'
      : item.label === 'real'
        ? 'wire__badge'
        : 'wire__badge wire__badge--unknown';
    const badgeText = item.label === 'fake' ? 'FAKE' : item.label === 'real' ? 'REAL' : '—';

    li.innerHTML = `
      <span class="${badgeClass}">${badgeText}</span>
      <div class="wire__body">
        <a class="wire__headline" href="${item.link || '#'}" target="_blank" rel="noopener noreferrer">${item.title || 'Untitled'}</a>
        <div class="wire__meta">${item.source || 'Unknown source'}${item.confidence != null ? ' · ' + item.confidence + '% confidence' : ''}</div>
      </div>
    `;
    wireList.appendChild(li);
  }

  const ageSeconds = updatedTs ? (Date.now() / 1000) - updatedTs : null;
  wireUpdated.textContent = ageSeconds != null ? `updated ${timeAgo(ageSeconds)}` : '';
}

async function pollWire() {
  try {
    const res = await fetch('/live_feed');
    const data = await res.json();
    renderWire(data.items, data.updated);
  } catch (err) {
    wireUpdated.textContent = 'offline';
  }
}

pollWire();
setInterval(pollWire, 30000); // re-poll every 30s; server caches for LIVE_REFRESH_SECONDS
