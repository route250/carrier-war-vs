"use strict";

// ロビー/ログイン周り + 共有のAPP/DOM/ユーティリティ

// 共有状態（プレイ画面からも利用）
window.APP = window.APP || {
  view: "entrance",
  username: "",
  match: null,
  matchSSE: null,
  matchHex: null,
  matchPending: { carrier_target: null, launch_target: null },
  matchLoggedUpToTurn: 0,
  matchHover: null,
};
APP.aiMon = APP.aiMon || { open: false, items: [] };

// DOM参照（プレイ側でも利用するため共有）
window.el = window.el || {
  // Entrance/Lobby
  viewEntrance: document.getElementById("view-entrance"),
  viewLobby: document.getElementById("view-lobby"),
  // Match
  viewMatch: document.getElementById("view-match"),
  usernameInput: document.getElementById("usernameInput"),
  enterLobby: document.getElementById("enterLobby"),
  lobbyUser: document.getElementById("lobbyUser"),
  // PvP lobby
  lobbyPvp: document.getElementById("lobby-pvp"),
  btnCreateMatch: document.getElementById("btnCreateMatch"),
  matchList: document.getElementById("matchList"),
  // Match room
  matchInfo: document.getElementById("matchInfo"),
  btnLeaveMatch: document.getElementById("btnLeaveMatch"),
  btnSubmitReady: document.getElementById("btnSubmitReady"),
  matchCanvas: document.getElementById("matchCanvas"),
  matchCarrierStatus: document.getElementById("matchCarrierStatus"),
  matchSquadronList: document.getElementById("matchSquadronList"),
  matchLog: document.getElementById("matchLog"),
  matchHint: document.getElementById("matchHint"),
  btnMatchModeMove: document.getElementById("btnMatchModeMove"),
  btnMatchModeLaunch: document.getElementById("btnMatchModeLaunch"),
};

// 共通ユーティリティ
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

// 画面切替
function showView(name) {
  APP.view = name;
  el.viewEntrance && el.viewEntrance.classList.toggle("hidden", name !== "entrance");
  el.viewLobby && el.viewLobby.classList.toggle("hidden", name !== "lobby");
  el.viewMatch && el.viewMatch.classList.toggle("hidden", name !== "match");
  if (name === "lobby") {
    try { startLobbySSE(); } catch {}
  }
}

// Lobby SSE
let LOBBY_SSE = null;
function startLobbySSE() {
  stopLobbySSE();
  try {
    const es = new EventSource('/v1/match/events');
    LOBBY_SSE = es;
    es.onmessage = (ev) => {
      try {
        const payload = JSON.parse(ev.data);
        if (payload && Array.isArray(payload.matches)) {
          renderMatchList(payload.matches);
        }
      } catch {}
    };
    es.addEventListener('list', (ev) => {
      try { const js = JSON.parse(ev.data); renderMatchList(js.matches || []); } catch {}
    });
    es.onerror = () => {
      handleSSEDisconnect('lobby');
    };
  } catch (e) {
    if (el.matchList) el.matchList.textContent = 'ロビーSSE初期化エラー';
  }
}
function stopLobbySSE() { try { if (LOBBY_SSE) LOBBY_SSE.close(); } catch {}; LOBBY_SSE = null; }

// SSE切断時の共通ハンドラ
function handleSSEDisconnect(context) {
  try { if (typeof stopMatchSSE === 'function') stopMatchSSE(); } catch {}
  try { stopLobbySSE(); } catch {}
  try {
    if (APP.match) {
      fetch(`/v1/match/${APP.match.id}/leave?token=${encodeURIComponent(APP.match.token)}`, { method: 'POST' }).catch(()=>{});
    }
  } catch {}
  APP.match = null;
  try { alert('通信がきれました'); } catch {}
  showView('entrance');
}

// マッチ一覧
async function refreshMatchList() {
  try {
    const res = await fetch('/v1/match/');
    if (!res.ok) throw new Error(`status ${res.status}`);
    const json = await res.json();
    renderMatchList(json.matches || []);
  } catch (e) {
    if (el.matchList) el.matchList.textContent = '取得に失敗しました';
  }
}

function renderMatchList(matches) {
  if (!el.matchList) return;
  if (!matches.length) { el.matchList.classList.add('empty'); el.matchList.textContent = '参加待ちのマッチはありません'; return; }
  el.matchList.classList.remove('empty');
  el.matchList.innerHTML = matches.map(m => {
    const open = m.has_open_slot && m.status !== 'over';
    const names = (Array.isArray(m.players) && m.players.length) ? escapeHtml(m.players.join(' vs ')) : '---';
    return `<div class="match-card">
      <div>
        <div class="mono">${m.match_id.slice(0,8)}</div>
        <div class="meta">${names}</div>
        <div class="meta">${m.status} ・ turn? ・ ${open ? '募集中' : '満席'}</div>
      </div>
      <div>
        <button data-join="${m.match_id}" ${open ? '' : 'disabled'}>参加</button>
      </div>
    </div>`;
  }).join('');
  el.matchList.querySelectorAll('button[data-join]').forEach(btn => {
    btn.addEventListener('click', () => joinMatch(btn.getAttribute('data-join')));
  });
}

// マッチ作成/参加
async function createMatch() {
  const name = APP.username || 'Player';
  try {
    const res = await fetch('/v1/match/', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: 'pvp', display_name: name }) });
    if (!res.ok) throw new Error(`status ${res.status}`);
    const json = await res.json();
    APP.match = { id: json.match_id, token: json.player_token, side: json.side, mode: json.mode || 'pvp' };
    if (typeof window.openMatchRoom === 'function') {
      window.openMatchRoom();
    } else {
      //try { if (typeof window.showView === 'function') window.showView('match'); } catch {}
    }
  } catch (e) {
    alert('マッチ作成に失敗しました');
  }
}

async function joinMatch(matchId) {
  const name = APP.username || 'Player';
  try {
    const res = await fetch(`/v1/match/${matchId}/join`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ display_name: name }) });
    if (!res.ok) throw new Error(`status ${res.status}`);
    const json = await res.json();
    APP.match = { id: json.match_id, token: json.player_token, side: json.side, mode: 'pvp' };
    if (typeof window.openMatchRoom === 'function') {
      window.openMatchRoom();
    } else {
      //try { if (typeof window.showView === 'function') window.showView('match'); } catch {}
    }
  } catch (e) {
    alert('参加に失敗しました');
    refreshMatchList();
  }
}

// LLMメニュー関連
async function fetchAICatalogForMenus() {
  try {
    const res = await fetch('/v1/match/ai');
    const ok = res && res.ok;
    const json = ok ? await res.json() : { providers: [] };
    const providers = Array.isArray(json.providers) ? json.providers : [];
    const byName = new Map();
    for (const p of providers) {
      const k = String(p?.name || '').toLowerCase();
      byName.set(k, Array.isArray(p?.models) ? p.models : []);
    }
    renderLLMProviderMenu('cpu', byName.get('cpu'));
    renderLLMProviderMenu('openai', byName.get('openai'));
    renderLLMProviderMenu('anthropic', byName.get('anthropic'));
    renderLLMProviderMenu('gemini', byName.get('gemini'));
  } catch (e) {
    renderLLMProviderMenu('openai', [{ name: 'gpt-4o-mini', model: 'gpt-4o-mini' }]);
  }
}

function renderLLMProviderMenu(provider, models) {
  const menu = document.getElementById(`llmMenu_${provider}`);
  if (!menu) return;
  const arr = Array.isArray(models) ? models : [];
  if (!arr.length) { menu.innerHTML = '<div class="muted">モデル一覧なし</div>'; return; }
  const rows = arr.map(m => {
    const id = m && (m.model || m.name);
    const label = m && (m.name || id);
    if (!id) return '';
    return `<button data-model="${id}" role="menuitem">${escapeHtml(label)}</button>`;
  }).join('');
  menu.innerHTML = `<div class="menu-group"><div class="menu-items">${rows}</div></div>`;
}

function wireLLMProvider(provider) {
  const dropdown = document.getElementById(`llmDropdown_${provider}`);
  const btn = document.getElementById(`btnLLM_${provider}`);
  const menu = document.getElementById(`llmMenu_${provider}`);
  if (!dropdown || !btn || !menu) return;
  const hide = () => menu.classList.add('hidden');
  const toggle = () => menu.classList.toggle('hidden');
  btn.addEventListener('click', (e) => { e.stopPropagation(); toggle(); });
  menu.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-model]');
    if (!b) return;
    const model = b.getAttribute('data-model') || '';
    hide();
    if (!model) return;
    createProviderMatch(provider, model);
  });
  document.addEventListener('click', (e) => { try { if (!dropdown.contains(e.target)) hide(); } catch {} });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hide(); });
}

function wireLLMMenus() {
  ['cpu','openai','anthropic','gemini'].forEach(wireLLMProvider);
}

function createProviderMatch(provider, model) {
  const p = String(provider || '').toLowerCase();
  if (p === 'cpu' || p === 'rule') {
    const m = String(model||'').toLowerCase();
    const diff = m === 'easy' ? 'Easy' : m === 'hard' ? 'Hard' : 'Normal';
    return createLLMMatch('CPU', diff);
  }
  if (p === 'openai') return createLLMMatch('OpenAI', model);
  if (p === 'anthropic') return createLLMMatch('Anthropic', model);
  if (p === 'gemini') return createLLMMatch('Gemini', model);
  alert(`未対応のプロバイダです: ${provider}`);
}

async function createLLMMatch(provider, model) {
  const name = APP.username || 'Player';
  try {
    const body = { mode: 'pve', display_name: name, config: { provider: provider, llm_model: model } };
    const res = await fetch('/v1/match/', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!res.ok) throw new Error(`status ${res.status}`);
    const json = await res.json();
    APP.match = { id: json.match_id, token: json.player_token, side: json.side, mode: json.mode || 'pve' };
    if (typeof window.openMatchRoom === 'function') {
      window.openMatchRoom();
    } else {
      //try { if (typeof window.showView === 'function') window.showView('match'); } catch {}
    }
  } catch (e) {
    alert('LLM対戦の作成に失敗しました');
  }
}

// 初期化
function initLobbyApp() {
  try { const saved = localStorage.getItem('cw_username'); if (saved) APP.username = saved; } catch {}
  if (APP.username && el.usernameInput) el.usernameInput.value = APP.username;
  showView('entrance');

  // Entrance events
  el.enterLobby && el.enterLobby.addEventListener('click', () => {
    const name = (el.usernameInput?.value || '').trim();
    if (!name) { alert('ユーザ名を入力してください'); return; }
    APP.username = name;
    try { localStorage.setItem('cw_username', name); } catch {}
    if (el.lobbyUser) el.lobbyUser.textContent = `ユーザ: ${APP.username}`;
    showView('lobby');
  });
  el.usernameInput && el.usernameInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') el.enterLobby?.click(); });

  // Lobby actions
  el.btnCreateMatch && el.btnCreateMatch.addEventListener('click', createMatch);
  try { startLobbySSE(); } catch {}
  try { wireLLMMenus(); fetchAICatalogForMenus(); } catch {}
}

window.addEventListener('DOMContentLoaded', initLobbyApp);
