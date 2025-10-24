"use strict";

// プレイ画面（マッチルーム）用

const { makeHexRenderer, getCss } = HexMap;
const SQUAD_MAX_HP = 40;
const CARRIER_MAX_HP = 100;

// 依存する共有オブジェクト
const APP = window.APP || (window.APP = {});
const el = window.el || (window.el = {});
Object.assign(el, {
  viewMatch: document.getElementById('view-match'),
  matchInfo: document.getElementById('matchInfo'),
  btnLeaveMatch: document.getElementById('btnLeaveMatch'),
  btnSubmitReady: document.getElementById('btnSubmitReady'),
  matchCanvas: document.getElementById('matchCanvas'),
  matchCarrierStatus: document.getElementById('matchCarrierStatus'),
  matchSquadronList: document.getElementById('matchSquadronList'),
  matchLog: document.getElementById('matchLog'),
  matchHint: document.getElementById('matchHint'),
  btnMatchModeMove: document.getElementById('btnMatchModeMove'),
  btnMatchModeLaunch: document.getElementById('btnMatchModeLaunch'),
  matchCarrierStatusB: document.getElementById('matchCarrierStatusB'),
  matchSquadronListB: document.getElementById('matchSquadronListB'),
  matchHintB: document.getElementById('matchHintB'),
  btnSubmitReadyB: document.getElementById('btnSubmitReadyB'),
  btnMatchModeMoveB: document.getElementById('btnMatchModeMoveB'),
  btnMatchModeLaunchB: document.getElementById('btnMatchModeLaunchB'),
});

const MATCH_UI = {
  A: {
    sidebar: document.querySelector('.match-sidebar:not(.match-sidebar-b)'),
    matchCarrierStatus: document.getElementById('matchCarrierStatus'),
    matchSquadronList: document.getElementById('matchSquadronList'),
    matchHint: document.getElementById('matchHint'),
    btnSubmitReady: document.getElementById('btnSubmitReady'),
    btnMatchModeMove: document.getElementById('btnMatchModeMove'),
    btnMatchModeLaunch: document.getElementById('btnMatchModeLaunch'),
  },
  B: {
    sidebar: document.querySelector('.match-sidebar-b'),
    matchCarrierStatus: document.getElementById('matchCarrierStatusB'),
    matchSquadronList: document.getElementById('matchSquadronListB'),
    matchHint: document.getElementById('matchHintB'),
    btnSubmitReady: document.getElementById('btnSubmitReadyB'),
    btnMatchModeMove: document.getElementById('btnMatchModeMoveB'),
    btnMatchModeLaunch: document.getElementById('btnMatchModeLaunchB'),
  },
};

APP.activeUISide = (typeof APP.activeUISide === 'string') ? APP.activeUISide : 'A';
let MATCH_MODE = typeof window.MATCH_MODE === 'string' ? window.MATCH_MODE : 'move';
window.MATCH_MODE = MATCH_MODE;

function getActiveUISide() {
  return (APP.activeUISide === 'B') ? 'B' : 'A';
}

function computeDesiredUISide() {
  const match = APP.match || {};
  const mode = String(match.mode || APP.matchState?.mode || '').toLowerCase();
  const side = String(match.side || '').toUpperCase();
  if (mode === 'pvp' && side === 'B') return 'B';
  return 'A';
}

function applyMatchSidebarLayout(side) {
  const wantB = side === 'B';
  const hasB = !!(MATCH_UI.B?.matchCarrierStatus && MATCH_UI.B?.btnSubmitReady);
  const next = wantB && hasB ? 'B' : 'A';
  APP.activeUISide = next;
  const root = document.documentElement;
  root.style.setProperty('--sidea-width', next === 'A' ? '1fr' : '0px');
  root.style.setProperty('--sideb-width', next === 'B' ? '1fr' : '0px');
  const active = MATCH_UI[next];
  const inactive = MATCH_UI[next === 'A' ? 'B' : 'A'];
  if (active?.sidebar) active.sidebar.setAttribute('data-collapsed', 'false');
  if (inactive?.sidebar) inactive.sidebar.setAttribute('data-collapsed', 'true');
  if (inactive?.btnMatchModeMove) inactive.btnMatchModeMove.classList.remove('active');
  if (inactive?.btnMatchModeLaunch) inactive.btnMatchModeLaunch.classList.remove('active');
  if (inactive?.matchHint) inactive.matchHint.textContent = '';
  if (inactive?.btnMatchModeMove) inactive.btnMatchModeMove.disabled = true;
  if (inactive?.btnMatchModeLaunch) inactive.btnMatchModeLaunch.disabled = true;
  if (inactive?.btnSubmitReady) inactive.btnSubmitReady.disabled = true;
  el.matchCarrierStatus = active?.matchCarrierStatus || null;
  el.matchSquadronList = active?.matchSquadronList || null;
  el.matchHint = active?.matchHint || null;
  el.btnSubmitReady = active?.btnSubmitReady || null;
  el.btnMatchModeMove = active?.btnMatchModeMove || null;
  el.btnMatchModeLaunch = active?.btnMatchModeLaunch || null;
}

function ensureMatchSidebarSide(force = false) {
  const desired = computeDesiredUISide();
  if (force || getActiveUISide() !== desired) {
    applyMatchSidebarLayout(desired);
    setMatchMode(MATCH_MODE);
    return true;
  }
  return false;
}

function clearMatchLog() {
  if (el.matchLog) el.matchLog.innerHTML = '';

}
function matchLogMsg(msg, opts) {
  if (!el.matchLog) return;
  const prefix = (opts && typeof opts.turn === 'number' && typeof opts.step === 'number')
    ? `${opts.turn}-${opts.step}`
    : new Date().toLocaleTimeString('ja-JP', { hour12: false });
  const line = document.createElement('div');
  line.className = 'entry';
  line.innerHTML = `<span class="ts">[${prefix}]</span>${(window.escapeHtml||((s)=>String(s)))(String(msg))}`;
  el.matchLog.appendChild(line);
  el.matchLog.scrollTop = el.matchLog.scrollHeight;
}

function unit_to_label(name) {
    if (typeof name === 'string' ) {
        name = name.replace(/^[AB]/, '');
        name = name.replace(/\d+$/, '');
        if( name === 'C' ) return '敵空母';
        if( name === 'SQ' ) return '敵編隊';
    }
    return '敵部隊';
}

function formatTurnLog(tlog) {
  try {
    if (!tlog || typeof tlog !== 'object') return '';
    // schemas.py の TurnLog をログ文として整形
    // 参考: server/services/turn.py の turn_forward() 内での生成箇所
    const uid = String(tlog.unit_id || '?');
    const pos = (p) => (p && typeof p.x === 'number' && typeof p.y === 'number') ? `(${p.x},${p.y})` : '';
    const tid = tlog.target_id ? String(tlog.target_id) : null;
    const upos = pos(tlog.unit_pos);
    const tpos = pos(tlog.target_pos);
    const val = (typeof tlog.value === 'number') ? tlog.value : null;
    const extra = Array.isArray(tlog.logs) && tlog.logs.length ? ` | ${tlog.logs.join(' | ')}` : '';
    switch (tlog.report) {
      case 'target':
        // 発艦または目標変更
        if (tid && tpos) return `${uid}${upos}: ${unit_to_label(tid)}${tpos}を発見！攻撃に向かう ${extra}`;
        if (tpos) return `${uid}${upos}: 航空部隊の発艦完了。目的地:${tpos} ${extra}`;
        return `${uid}${upos}: 進路変更、目的地点:${extra}`;
      case 'engaging':
        return `${uid}${upos}: ${unit_to_label(tid)} と交戦開始 ${extra}`;
      case 'attack':
        if (uid && uid[1] === 'C') {
          return `${uid}${upos}: ${unit_to_label(tid)}に対空攻撃 ${val ?? 0}機を撃墜した。${extra}`;
        } else {
          return `${uid}${upos}: ${unit_to_label(tid)}を攻撃。ダメージ${val ?? 0}を与えた。 ${extra}`;
        }
      case 'hit':
        if (uid && uid[1] === 'C') {
          return `${unit_to_label(tid)}からの攻撃。${uid}${upos}にダメージ${val ?? 0}を受けた ${extra}`;
        } else {
          return `${unit_to_label(tid)}からの攻撃。${uid}${upos}の${val ?? 0}機が撃墜された。 ${extra}`;
        }
      case 'lost':
        // 撃墜/撃沈
        if (uid && uid[1] === 'C') {
          return `${unit_to_label(tid)}からの攻撃。${uid}${upos}が撃沈されました ${extra}`;
        } else {
          return `${unit_to_label(tid)}からの攻撃。${uid}${upos}が全滅 ${extra}`;
        }
      case 'returning':
        return `${uid}${upos}: 帰投に移る。${extra}`;
      case 'landed':
        return `${uid}${upos}: 空母 ${tid || ''} に着艦${extra}`.replace(/\s+$/,'');
      case 'found':
        return `${uid}${upos}: ${unit_to_label(tid)}${tpos} を発見${extra}`;
      default:
        return `${uid}${upos}: ${tlog.report}${extra}`;
    }
  } catch {}
}

// 発艦ターゲットをクライアント側でクリアし、UI とログを更新するユーティリティ
function clearPendingLaunchTarget(reason) {
  try {
    if (APP.matchPending && APP.matchPending.launch_target) {
      APP.matchPending.launch_target = null;
      if (reason) matchLogMsg(`[info] 発艦ターゲットをクリア: ${reason}`);
      try { renderMatchView(); updateMatchPanels(); updateMatchControls(); } catch {}
    }
  } catch {}
}

// 画面切替（Safariのグローバル衝突回避のため別名ラッパ）
function goView(name) {
  window.showView(name);
}

function openMatchRoom() {
  // ロビーSSE停止
  window.stopLobbySSE();
  goView('match');
  APP.aiMon = { open: false, items: [] };
  const root = document.documentElement;
  root.style.setProperty('--inspector-width', '0px');
  const inspector = document.getElementById('aiInspector');
  if (inspector) inspector.classList.add('hidden');
  ensureMatchSidebarSide(true);
  clearMatchLog();
  APP.matchLoggedUpToTurn = 0;
  // ゲームSSE開始
  startMatchSSE();
}

async function leaveMatchToLobby() {
  // 先にサーバへ離脱通知を送り、その後にSSEを閉じることで404ログを減らす
  if (APP.match && APP.match.id && APP.match.token) {
    await fetch(`/v1/match/${APP.match.id}/leave?token=${encodeURIComponent(APP.match.token)}`, { method: 'POST' }).catch(()=>{});
  }
  stopMatchSSE();
  APP.match = null;
  APP.activeUISide = 'A';
  applyMatchSidebarLayout('A');
  setMatchMode(MATCH_MODE);
  const root = document.documentElement;
  root.style.setProperty('--inspector-width', '0px');
  const inspector = document.getElementById('aiInspector');
  if (inspector) inspector.classList.add('hidden');
  goView('lobby');
  window.startLobbySSE();
}

// SSE（マッチ）
function startMatchSSE() {
  stopMatchSSE();
  if (!APP.match) return;
  try {
    const url = `/v1/match/${APP.match.id}/events?token=${encodeURIComponent(APP.match.token)}`;
    const es = new EventSource(url);
    APP.matchSSE = es;
    es.onmessage = (ev) => {
      try {
        const payload = JSON.parse(ev.data);
        if (payload && payload.type === 'state') {
          handleMatchStateUpdate(payload);
        }
      } catch {}
    };
    es.addEventListener('state', (ev) => {
      try { const js = JSON.parse(ev.data); handleMatchStateUpdate(js); } catch {}
    });
    es.onerror = () => { if (typeof window.handleSSEDisconnect === 'function') window.handleSSEDisconnect('match'); };
    if (el.matchInfo) el.matchInfo.textContent = '接続中…';
  } catch (e) {
    if (el.matchInfo) el.matchInfo.textContent = 'SSE初期化エラー';
  }
}

function stopMatchSSE() {
  try {
    if (APP.matchSSE) APP.matchSSE.close();
  } catch {}
  APP.matchSSE = null; 
}

// 描画
function renderMatchView() {
  const st = APP.matchState; const cv = el.matchCanvas; if (!st || !cv) return;
  const W = st.map_w || 30, H = st.map_h || 30;
  if (st.map) APP.matchMap = st.map;
  const getTile = (x,y) => { const m = APP.matchMap; if (!m) return 0; const row = m[y]; if (!row) return 0; const v = row[x]; return v||0; };
  if (!APP.matchHex || APP.matchHex.W !== W || APP.matchHex.H !== H || APP.matchHex.canvas !== cv) {
    APP.matchHex = makeHexRenderer(cv, W, H, getTile);
  } else { APP.matchHex.getTileFn = getTile; }

  APP.matchHex.renderBackground();
  // 可視範囲（サーバ提供）
  try {
    const mineObj = st && st.units;
    const visList = (mineObj && Array.isArray(mineObj.turn_visible)) ? mineObj.turn_visible : [];
    const visSet = new Set();
    for (const v of visList) {
      if (typeof v === 'string') {
        const m = v.match(/(-?\d+)\s*,\s*(-?\d+)/);
        if (m) { visSet.add(`${parseInt(m[1],10)},${parseInt(m[2],10)}`); continue; }
        const m2 = v.match(/x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)/i);
        if (m2) { visSet.add(`${parseInt(m2[1],10)},${parseInt(m2[2],10)}`); continue; }
      } else if (v && typeof v === 'object' && typeof v.x === 'number' && typeof v.y === 'number') {
        visSet.add(`${v.x},${v.y}`);
      }
    }
    if (visSet.size > 0 && APP.matchHex.renderVisibilityOverlay) {
      APP.matchHex.renderVisibilityOverlay(visSet);
    }
  } catch {}

  // 自軍空母
  const me = (st && st.units && st.units.carrier) || {};
  if (me && me.x!=null && me.y!=null) {
    APP.matchHex.drawCarrier(me.x, me.y, getCss('--carrier')||'#4aa3ff', me.hp, me.max_hp);
    // 空母の進路プレビュー
    const pending = APP.matchPending?.carrier_target;
    const srvTarget = me && me.target;
    if (pending && pending.x!=null && pending.y!=null && (pending.x !== me.x || pending.y !== me.y)) {
      APP.matchHex.drawLine(me.x, me.y, pending.x, pending.y, 'rgba(106,212,255,0.65)');
    } else if (srvTarget && srvTarget.x!=null && srvTarget.y!=null && (srvTarget.x !== me.x || srvTarget.y !== me.y)) {
      APP.matchHex.drawLine(me.x, me.y, srvTarget.x, srvTarget.y, 'rgba(106,212,255,0.25)');
    }
  }
  // 自軍編隊
  const mySqs = st.units && st.units.squadrons;
  if (Array.isArray(mySqs)) {
    for (const s of mySqs) {
      if (!s || s.state === 'onboard' || s.state === 'lost') continue;
      if (s.x == null || s.y == null) continue;
      APP.matchHex.drawSquadron(s.x, s.y, getCss('--squad')||'#f2c14e', s.hp, s.max_hp);
      if (s.target && s.target.x!=null && s.target.y!=null) {
        APP.matchHex.drawLine(s.x, s.y, s.target.x, s.target.y, 'rgba(242,193,78,0.25)');
      }
    }
  }
  // 敵空母（敵は記憶表示にスタイル切替）
  const op = (st && st.intel && st.intel.carrier) || {};
  if (op && op.x != null && op.y != null) {
    const enemyColor = getCss('--enemy')||'#ff6464';
    const key = `${op.x},${op.y}`;
    APP.matchHex.drawCarrier(op.x, op.y, enemyColor, op.hp, op.max_hp);
  }
  // 敵編隊（今ターン視認できたもの）
  const oppSqs = st.intel && st.intel.squadrons;
  if (Array.isArray(oppSqs)) {
    for (const s of oppSqs) {
      if (!s || s.x == null || s.y == null) continue;
      const enemyColor = getCss('--enemy')||'#ff6464';
      const key = `${s.x},${s.y}`;
      APP.matchHex.drawSquadron(s.x, s.y, enemyColor, s.hp, s.max_hp);
      if (s.x0 != null && s.y0 != null) {
        if (s.x0 !== s.x || s.y0 !== s.y) APP.matchHex.drawLine(s.x0, s.y0, s.x, s.y, 'rgba(255,255,255,0.65)');
      }
    }
  }
  // 発艦ターゲットのプレビュー
  const pendingLaunch = APP.matchPending?.launch_target;
  if (pendingLaunch && me && me.x!=null && me.y!=null) {
    APP.matchHex.drawLine(me.x, me.y, pendingLaunch.x, pendingLaunch.y, 'rgba(242,193,78,0.5)');
  }
  // ホバー枠
  try {
    APP.matchHex.renderHoverOutline({ mode: MATCH_MODE, hover: APP.matchHover, carrier: me, squadrons: st.units?.squadrons });
  } catch {}
}

// 操作可否
function updateMatchControls() {
  const s = APP.matchState || {};
  const status = s.status;
  const wait = s.waiting_for;
  const disableAll = !APP.match || !s || status !== 'active';
  const canSubmit = !disableAll && (wait === 'you' || wait === 'orders');
  const canEdit = !disableAll && (wait === 'you' || wait === 'orders');
  const sqArr = APP.matchState?.units?.squadrons || [];
  const launchable = Array.isArray(sqArr) ? sqArr.filter(x => x && x.state === 'onboard' && (x.hp ?? 0) > 0).length > 0 : false;
  const activeSide = getActiveUISide();
  for (const [side, ui] of Object.entries(MATCH_UI)) {
    if (!ui) continue;
    const isActive = side === activeSide;
    if (ui.btnMatchModeMove) {
      ui.btnMatchModeMove.disabled = disableAll || !canEdit || !isActive;
      ui.btnMatchModeMove.classList.toggle('active', isActive && !disableAll && canEdit && MATCH_MODE === 'move');
    }
    if (ui.btnMatchModeLaunch) {
      ui.btnMatchModeLaunch.disabled = disableAll || !canEdit || !launchable || !isActive;
      ui.btnMatchModeLaunch.classList.toggle('active', isActive && !disableAll && canEdit && MATCH_MODE === 'launch');
    }
    if (ui.btnSubmitReady) ui.btnSubmitReady.disabled = disableAll || !canSubmit || !isActive;
    if (!isActive && ui.matchHint) ui.matchHint.textContent = '';
  }
}

// キャンバス操作
function setMatchMode(m) {
  if (m === 'launch') {
    const sqArr = APP.matchState?.units?.squadrons || [];
    const launchable = Array.isArray(sqArr) ? sqArr.filter(x => x && x.state === 'onboard' && (x.hp ?? 0) > 0).length > 0 : false;
    if (!launchable) {
      const activeUI = MATCH_UI[getActiveUISide()];
      if (activeUI?.matchHint) activeUI.matchHint.textContent = '発艦可能な編隊がありません';
      return;
    }
  }
  MATCH_MODE = m;
  window.MATCH_MODE = MATCH_MODE;
  let hintText = '';
  if (m === 'move') {
    hintText = '目的地をクリック';
  } else if (m === 'launch') {
    const sqs = APP.matchState?.units?.squadrons || [];
    const bases = Array.isArray(sqs) ? sqs.filter(x => x && x.state === 'onboard' && (x.hp ?? 0) > 0) : [];
    const maxFuel = bases.length ? Math.max(...bases.map(x => (typeof x.fuel === 'number') ? x.fuel : 0)) : null;
    hintText = (maxFuel != null) ? `目標地点をクリック（航続距離:${maxFuel}）` : '発艦可能な編隊がありません';
  }
  for (const [side, ui] of Object.entries(MATCH_UI)) {
    if (!ui?.matchHint) continue;
    ui.matchHint.textContent = (side === getActiveUISide()) ? hintText : '';
  }
  updateMatchControls();
}

function onMatchCanvasClick(ev) {
  if (!APP.match || !APP.matchState) return;
  if (!APP.matchHex) return;
  const s = APP.matchState || {};
  const status = s.status;
  const canEdit = (status === 'active') && (s && (s.waiting_for === 'you' || s.waiting_for === 'orders'));
  const t = APP.matchHex.tileFromEvent(ev);
  if (!t) return;
  const me = s.units?.carrier;
  const sqs = s.units?.squadrons;
  if (MATCH_MODE === 'move') {
    if (!canEdit) return;
    if (!APP.matchHex.isValidTarget({ mode: 'move', x: t.x, y: t.y, carrier: me, squadrons: sqs })) return;
    APP.matchPending = { ...(APP.matchPending||{}), carrier_target: { x: t.x, y: t.y } };
    renderMatchView(); updateMatchPanels();
  } else if (MATCH_MODE === 'launch') {
    if (!canEdit) return;
    if (!APP.matchHex.isValidTarget({ mode: 'launch', x: t.x, y: t.y, carrier: me, squadrons: sqs })) return;
    APP.matchPending = { ...(APP.matchPending||{}), launch_target: { x: t.x, y: t.y } };
    renderMatchView(); updateMatchPanels();
  }
}
function onMatchCanvasMove(ev) { if (!APP.match || !APP.matchHex) return; APP.matchHover = APP.matchHex.tileFromEvent(ev); renderMatchView(); }
function onMatchCanvasLeave() { APP.matchHover = null; renderMatchView(); }

async function submitReady() {
  if (!APP.match) return;
  try {
    const s = APP.matchState || {};
    if (!(s && s.status === 'active')) return;
  } catch {}
  try {
    // 送信前に出撃可能編隊があるか確認。なければ launch_target を送らない
    const sqArr = APP.matchState?.units?.squadrons || [];
    const launchableCount = Array.isArray(sqArr) ? sqArr.filter(x => x && x.state === 'onboard' && (x.hp ?? 0) > 0).length : 0;
    const staged = {};
    if (APP.matchPending?.carrier_target) staged.carrier_target = APP.matchPending.carrier_target;
    if (launchableCount > 0 && APP.matchPending?.launch_target) staged.launch_target = APP.matchPending.launch_target;
    else {
      // クライアント側の pending も明示的にクリアしておく
      clearPendingLaunchTarget(launchableCount === 0 ? '出撃可能編隊がありません' : '送信前の自動クリア');
    }
    const res = await fetch(`/v1/match/${APP.match.id}/orders`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ player_token: APP.match.token, player_orders: staged }) });
    if (!res.ok) {
      // サーバエラーであれば、出撃可能数がゼロならクライアントの launch_target をクリアして UI 更新
  if (launchableCount === 0) clearPendingLaunchTarget('サーバ応答エラー');
      throw new Error(`status ${res.status}`);
    }
    const js = await res.json();
    if (js && js.accepted === false) {
      try {
        if (Array.isArray(js.logs)) { for (const m of js.logs) matchLogMsg(`[NG] ${m}`); }
        else matchLogMsg('[NG] 注文が受理されませんでした');
      } catch {}
      // サーバが注文を拒否した場合も、出撃可能数がゼロならクライアントの launch_target をクリア
  if (launchableCount === 0) clearPendingLaunchTarget('サーバが注文を拒否');
      return;
    }
  } catch (e) { alert('送信に失敗しました'); }
}

function handleMatchStateUpdate(nextState) {
  try {
    const prevTurn = (APP.matchState && typeof APP.matchState.turn === 'number') ? APP.matchState.turn : null;
    const nextTurn = (nextState && typeof nextState.turn === 'number') ? nextState.turn : null;
    if (prevTurn != null && nextTurn != null && nextTurn > prevTurn) {
      APP.matchPending = { carrier_target: null, launch_target: null };
    }
    if (typeof nextTurn === 'number' && nextTurn > (APP.matchLoggedUpToTurn || 0)) {
      if (Array.isArray(nextState.logs)) {
        for (const m of nextState.logs) {
          const step = (m && typeof m.step === 'number') ? m.step : null;
          matchLogMsg(formatTurnLog(m), (step!=null ? { turn: nextTurn, step } : undefined));
        }
      }
      APP.matchLoggedUpToTurn = nextTurn;
    }
  } catch {}
  APP.matchState = nextState;
  ensureMatchSidebarSide();
  try { if (nextState && nextState.ai_diag) onAIDiag(nextState.ai_diag); } catch {}
  try { updateAIIndicator(nextState); } catch {}
  renderMatchView();
  updateMatchPanels();
  updateMatchControls();
}

// AI インジケータ/インスペクタ
function aiOpponentPresent() { return Array.isArray(APP.aiMon?.items) && APP.aiMon.items.length > 0; }
function updateAIIndicator(state) {
  const btn = document.getElementById('btnAIInspector'); if (!btn) return;
  const waiting = state && state.waiting_for;
  const thinking = aiOpponentPresent() && state && state.status === 'active' && waiting === 'opponent';
  btn.classList.toggle('ai-thinking', !!thinking);
}
function toggleAIInspector() {
  APP.aiMon.open = !APP.aiMon.open;
  const pane = document.getElementById('aiInspector');
  const root = document.documentElement;
  const open = !!APP.aiMon.open;
  root.style.setProperty('--inspector-width', open ? '1fr' : '0px');
  if (!pane) return;
  pane.classList.toggle('hidden', !open);
  if (open) renderAIInspector();
}
function onAIDiag(diag) {
  try {
    if (!diag || typeof diag !== 'object') return;
    const key = `${diag.turn}-${diag.model}-${diag.ms}`;
    const idx = APP.aiMon.items.findIndex(it => it.key === key);
    const item = {
      key,
      turn: diag.turn,
      model: diag.model,
      ms: diag.ms,
      pt: diag.pt,
      ct: diag.ct,
      cost: diag.cost,
      total_cost: diag.total_cost,
      thinking: diag.thinking,
    };
    if (idx >= 0) APP.aiMon.items[idx] = item; else APP.aiMon.items.unshift(item);
    if (APP.aiMon.items.length > 20) APP.aiMon.items.length = 20;
    const btn = document.getElementById('btnAIInspector'); if (btn) { btn.classList.add('ai-pulse'); setTimeout(() => btn.classList.remove('ai-pulse'), 700); }
    try { if (APP.matchState) updateAIIndicator(APP.matchState); } catch {}
    if (APP.aiMon.open) renderAIInspector();
  } catch {}
}
function renderAIInspector() {
  const body = document.getElementById('aiInspectorBody');
  if (!body) return;
  if (!APP.aiMon.items.length) { body.innerHTML = '<div class="muted">AI thinking は未取得</div>'; return; }
  body.innerHTML = APP.aiMon.items.map(it => {
    const pt = (it.pt ?? '') !== '' ? ` pt:${it.pt}` : '';
    const ct = (it.ct ?? '') !== '' ? ` ct:${it.ct}` : '';
    const t = (typeof it.turn === 'number') ? `T${it.turn} · ` : '';
    const cost = (typeof it.cost === 'number') ? ` cost:$${it.cost.toFixed(4)}` : '';
    const tcost = (typeof it.total_cost === 'number') ? ` total:$${it.total_cost.toFixed(4)}` : '';
    const meta = `${t}${it.ms}ms${pt}${ct}${cost}${tcost ? ` (${tcost})` : ''}`;
    const model = (it.model || '').trim();
    return `<div class="ai-entry">
      <div class="meta mono">${model}｜${meta}</div>
      <div class="thinking">${(window.escapeHtml||((s)=>String(s)))(it.thinking || '')}</div>
    </div>`;
  }).join('');
}

function updateMatchPanels() {
  if (!APP.matchState || !el.matchCarrierStatus) return;
  const s = APP.matchState || {};
  const mineC = s.units?.carrier || {}, oppC = s.intel?.carrier || {};
  const mySide = (APP.match?.side === 'A') ? 'A' : 'B';
  const mine = mineC; const opp = oppC;
  const mySqs = s.units && s.units.squadrons;
  const sqArr = Array.isArray(mySqs) ? mySqs : [];
  const baseAvail = sqArr.filter((x) => x && x.state === 'onboard' && (x.hp ?? 0) > 0);
  const totalSlots = sqArr.length;
  let name = String(mine.id || '-');
  const hpNow = (typeof mine.hp === 'number') ? mine.hp : '-';
  const hpMax = CARRIER_MAX_HP;
  const onboard = `${baseAvail.length} / ${totalSlots || '-'}`;
  const hpLine = `${hpNow} / ${hpMax}`;
  const spd = (typeof mine.speed === 'number') ? `${mine.speed}` : '-';
  el.matchCarrierStatus.innerHTML = `
    <div class="kv">
      <div>${name}</div><div>HP ${hpLine}</div>
      <div>速度</div><div>${spd}</div>
      <div>航空部隊</div><div>${onboard}</div>
    </div>
  `;
  if (el.matchSquadronList) {
    const arr = sqArr;
    if (arr.length === 0) {
      el.matchSquadronList.textContent = '編隊はありません';
    } else {
      const rows = arr.map((sq) => {
        const st = sq.state || '-';
        const hpNow = (typeof sq.hp === 'number') ? sq.hp : '-';
        const hpMax = SQUAD_MAX_HP;
        let name = String(sq.id || '-');
        const m = name.match(/(SQ\d+)$/); if (m) name = m[1];
        const spd = (typeof sq.speed === 'number') ? sq.speed : '-';
        const fuel = (typeof sq.fuel === 'number') ? sq.fuel : '-';
        const line1 = `<div class="kv"><div class="mono">${name}</div><div>HP ${hpNow} / ${hpMax}</div></div>`;
        const line2 = `<div class="kv"><div>状態</div><div>${st}</div></div>`;
        const line3 = `<div class="kv"><div>速度/航続</div><div>${spd} / ${fuel}</div></div>`;
        return `${line1}${line2}${line3}`;
      }).join('');
      el.matchSquadronList.innerHTML = `<div class="list">${rows}</div>`;
    }
  }
  try {
    // ヘッダ表示
    if (s.status === 'waiting') {
      if (el.matchInfo) el.matchInfo.textContent = '参加者待ち…';
    } else if (s.status === 'active') {
      const oppName = s.opponent_name || '???';
      const turnNow = (typeof s.turn === 'number') ? s.turn : null;
      const turnMax = (typeof s.max_turn === 'number') ? s.max_turn : null;
      const turnStr = (turnNow != null && turnMax != null) ? `${turnNow}/${turnMax}` : '-/-';
      let phase = 'ターン解決中';
      switch (s.waiting_for) {
        case 'you': phase = 'あなたのオーダー待ち'; break;
        case 'opponent': phase = '相手のオーダー待ち'; break;
        case 'orders': phase = 'オーダー受付中'; break;
        default: phase = 'ターン解決中';
      }
      if (el.matchInfo) el.matchInfo.textContent = `相手: ${oppName} Turn: ${turnStr} ${phase}`;
    } else if (s.status === 'over') {
      let result = s.result || null;
      if (!result) {
        const myHp = mine && typeof mine.hp === 'number' ? mine.hp : null;
        const opHp = opp && typeof opp.hp === 'number' ? opp.hp : null;
        if (myHp != null && opHp != null) {
          if (myHp <= 0 && opHp <= 0) result = 'draw';
          else if (myHp <= 0) result = 'lose';
          else if (opHp <= 0) result = 'win';
        }
      }
      const endText = result === 'win' ? 'ゲーム終了（あなたの勝ち）'
                    : result === 'lose' ? 'ゲーム終了（あなたの負け）'
                    : 'ゲーム終了（引き分け）';
      const turnNow = (typeof s.turn === 'number') ? s.turn : null;
      const turnMax = (typeof s.max_turn === 'number') ? s.max_turn : null;
      const turnStr = (turnNow != null && turnMax != null) ? `${turnNow}/${turnMax}` : '-/-';
      if (el.matchInfo) el.matchInfo.textContent = `${endText} Turn: ${turnStr}`;
    }
  } catch (e) {}
}

// 初期イベント配線（プレイ画面）
function initPlayBindings() {
  document.getElementById('btnAIInspector')?.addEventListener('click', toggleAIInspector);
  document.getElementById('btnAIInspectorClose')?.addEventListener('click', toggleAIInspector);
  el.btnLeaveMatch && el.btnLeaveMatch.addEventListener('click', leaveMatchToLobby);
  const submitButtons = [MATCH_UI.A?.btnSubmitReady, MATCH_UI.B?.btnSubmitReady].filter(Boolean);
  submitButtons.forEach((btn) => btn.addEventListener('click', submitReady));
  el.matchCanvas && el.matchCanvas.addEventListener('click', onMatchCanvasClick);
  el.matchCanvas && el.matchCanvas.addEventListener('mousemove', onMatchCanvasMove);
  el.matchCanvas && el.matchCanvas.addEventListener('mouseleave', onMatchCanvasLeave);
  const moveButtons = [MATCH_UI.A?.btnMatchModeMove, MATCH_UI.B?.btnMatchModeMove].filter(Boolean);
  moveButtons.forEach((btn) => btn.addEventListener('click', () => setMatchMode('move')));
  const launchButtons = [MATCH_UI.A?.btnMatchModeLaunch, MATCH_UI.B?.btnMatchModeLaunch].filter(Boolean);
  launchButtons.forEach((btn) => btn.addEventListener('click', () => setMatchMode('launch')));
}

window.addEventListener('DOMContentLoaded', initPlayBindings);

// エクスポート（グローバル）
window.openMatchRoom = openMatchRoom;
window.leaveMatchToLobby = leaveMatchToLobby;
window.startMatchSSE = startMatchSSE;
window.stopMatchSSE = stopMatchSSE;
window.updateMatchPanels = updateMatchPanels;
