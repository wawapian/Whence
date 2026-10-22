import { api, post, downloadBlob } from './api.js';
import { icon, hydrateIcons } from './icons.js';
import { esc, attr, date, bytes, badge, button, originalLink, empty, answerCard, sourcePanel } from './components.js';
import * as views from './views.js';
import { modelSettings } from './model-settings.js';

const $ = (selector, root = document) => root.querySelector(selector);
const state = {
  route: location.hash.slice(1) || 'chat', overview: {}, memories: [], documents: [],
  reminders: { due: [], pending: [], completed: [] }, trash: { records: [], documents: [] },
  history: { items: [], total: 0 }, historySearch: '', historyPeriod: 'all',
  messages: [], qaLog: [], draft: '', attachments: [], allowCloud: false, busy: null,
  fileFilter: 'all', fileSearch: '', cloudFilter: 'all', selected: new Set(),
  timelineFilter: 'all', timelineSearch: '', reminderFilter: 'active', trashFilter: 'all',
  memoryLimit: 50, chatLayout: 'split', listening: false,
};
let refreshInFlight = false, historySequence = 0, searchTimer, speech, lastDialogTrigger;
let uploadFiles = [], spendingData, spendingPeriod = 'this_month', spendingCurrency;
const dialog = $('#detail-dialog');
// UI 文案在模板处按模式选择，原件、答案与用户记录永不作全局文字替换。
function demoCopy(html) { return html; }

function toast(text, error = false) {
  const el = document.createElement('div'); el.className = `toast ${error ? 'error' : ''}`;
  el.innerHTML = `${icon(error ? 'info' : 'check')}<span>${esc(text)}</span>`;
  $('#toasts').append(el); setTimeout(() => el.remove(), error ? 7000 : 4500);
}
function showError(error) { toast(error.message || String(error), true); }
function setGlobalError(text) {
  const el = $('#global-error'); el.hidden = !text;
  el.innerHTML = text ? `${icon('info')}<span>${esc(text)}</span>${button('重试连接', 'refresh', { cls: 'subtle small', icon: 'refresh' })}` : '';
  $('#connection-label').textContent = text ? '服务连接异常' : '本地空间已连接';
  $('#connection-dot').classList.toggle('offline', !!text);
}
function nav() { $('#navigation').innerHTML = views.navigation(state); $('#breadcrumb-title').textContent = views.routes.find(([id]) => id === state.route)?.[1] || '对话工作台'; }
function closeMenu() { document.body.classList.remove('menu-open'); $('.menu-button').setAttribute('aria-expanded', 'false'); }
function render() {
  if (!views.routes.some(([id]) => id === state.route)) state.route = 'chat';
  nav(); $('#main').innerHTML = demoCopy(views[state.route](state)); hydrateIcons();
  document.title = `${views.routes.find(([id]) => id === state.route)[1]} · Whence · 来处`;
  closeMenu(); if (state.route === 'chat') scrollFeed();
}
function updateFeed() {
  if (!$('#chat-feed')) return;
  $('#chat-feed').innerHTML = demoCopy(views.feed(state));
  const clear = $('[data-action="clear-chat"]'); if (clear) clear.disabled = !state.messages.length;
  scrollFeed();
}
function scrollFeed() { requestAnimationFrame(() => { const feed = $('#chat-feed'); if (feed) feed.scrollTop = state.messages.length ? feed.scrollHeight : 0; }); }
function updateChatPanels() {
  if ($('#recent-panel')) $('#recent-panel').innerHTML = demoCopy(views.recentMemories(state));
  if ($('#reminder-strip-container')) $('#reminder-strip-container').innerHTML = demoCopy(views.reminderStrip(state));
  let changed = false;
  for (const message of state.messages) {
    if (message.kind !== 'record') continue;
    const memory = state.memories.find(m => m.raw_id === message.raw_id);
    const status = memory?.state === 'pending' ? '原文已保存。AI 整理暂不可用，可继续查看原始记录。' : memory?.fact_id ? '这条记录已完成整理，可在时间线查看结构化记忆。' : memory?.consolidated === 1 ? '原文已留存，未提取到结构化事实。可继续浏览原始记录。' : message.status;
    if (status !== message.status) { message.status = status; changed = true; }
  }
  if (changed) updateFeed();
}
async function refresh({ repaint = true, manual = false } = {}) {
  if (refreshInFlight) return; refreshInFlight = true;
  const previous = JSON.stringify([state.overview, state.memories, state.documents, state.reminders, state.trash]);
  const previousHistoryCount = state.overview.history;
  const requests = [
    ['overview', '/api/workspace/overview'], ['memories', `/api/memories?limit=${state.memoryLimit}`],
    ['documents', '/api/documents'], ['reminders', '/api/reminders'], ['trash', '/api/trash'],
  ];
  try {
    const results = await Promise.allSettled(requests.map(([, path]) => api(path)));
    const failures = [];
    results.forEach((r, i) => {
      if (r.status === 'rejected') { failures.push(r.reason.message); return; }
      const name = requests[i][0]; state[name] = name === 'documents' ? r.value.documents : r.value;
    });
    const available = new Set(state.documents.map(d => d.id)); state.selected.forEach(id => { if (!available.has(id)) state.selected.delete(id); });
    setGlobalError(failures.length ? [...new Set(failures)].join('；') : '');
    nav();
    if (repaint) {
      if (state.route === 'chat') updateChatPanels();
      else if ((manual || previous !== JSON.stringify([state.overview, state.memories, state.documents, state.reminders, state.trash])) &&
               state.route !== 'history' && !dialog.open && !document.activeElement?.matches('input,textarea,select')) {
        const scrollTop = $('#main').scrollTop;
        // Keep semantic results open while background file parsing changes the list.
        if (state.route === 'files' && $('#semantic-results')?.childElementCount) {
          $('#file-grid').innerHTML = views.fileGrid(state);
        } else { render(); $('#main').scrollTop = scrollTop; }
      }
    }
    if (manual && !failures.length) toast('已更新本地数据');
    if (state.route === 'history' && !dialog.open && (manual || previousHistoryCount !== state.overview.history)) await loadHistory(false);
  } finally { refreshInFlight = false; }
}
async function loadHistory(append = false) {
  const sequence = ++historySequence;
  const params = new URLSearchParams({ limit: '50', offset: append ? String(state.history.items.length) : '0', q: state.historySearch, period: state.historyPeriod });
  try {
    const result = await api(`/api/qa-history?${params}`); if (sequence !== historySequence) return;
    state.history = { items: append ? [...state.history.items, ...result.items] : result.items, total: result.total };
    if ($('#history-results')) $('#history-results').innerHTML = views.historyRows(state);
  } catch (error) { if (sequence === historySequence) showError(error); }
}
function goChat(text, focus = true) {
  if (text !== undefined) state.draft = text;
  if (state.route !== 'chat') { location.hash = '#chat'; }
  else render();
  if (focus) requestAnimationFrame(() => { $('#composer-text')?.focus(); });
}
function busyControls() {
  const composer = $('#composer'); if (!composer) return;
  composer.querySelectorAll('button, textarea, input').forEach(el => { el.disabled = !!state.busy; });
}
async function submit(mode, explicitQuestion) {
  if (state.busy) return;
  if (speech) speech.stop();
  const text = (explicitQuestion ?? state.draft).trim();
  if (!text && !state.attachments.length) { toast(mode === 'ask' ? '先写下你想问的问题。' : '写下一句话，或添加一份原件。'); $('#composer-text')?.focus(); return; }
  if (mode === 'ask' && state.attachments.length) { toast('请先记录并保存附件，整理完成后再提问。'); return; }
  if (mode === 'ask' && !text) { toast('请写下你想问的问题。'); return; }
  const files = [...state.attachments], allowCloud = state.allowCloud, start = performance.now();
  state.messages.push({ kind: 'user', mode, text, files: files.map(f => f.name), at: new Date() });
  state.busy = mode; updateFeed(); busyControls();
  try {
    if (mode === 'record') {
      const form = new FormData(); form.set('text', text); form.set('sensitive', String(!allowCloud));
      files.forEach(f => form.append('files', f));
      const result = await post('/api/memories', form);
      state.messages.push({ kind: 'record', raw_id: result.raw_id });
      state.attachments = [];
    } else {
      const result = await post('/api/query', { question: text, history: state.qaLog.slice(-10) });
      state.messages.push({ kind: 'answer', result, question: text, elapsed: ((performance.now() - start) / 1000).toFixed(2) });
      if (!result.status || result.status === 'ok') {
        const title = result.execution?.reading?.title;
        state.qaLog.push({ question: text, answer: result.answer + (title ? `\n关联文档：${title}` : '') });
        state.qaLog = state.qaLog.slice(-10);
      }
    }
    state.draft = ''; state.allowCloud = false;
    if ($('#composer-text')) $('#composer-text').value = '';
    if ($('#attachment-list')) $('#attachment-list').innerHTML = views.attachments(state);
    if ($('#allow-cloud')) $('#allow-cloud').checked = false;
    await refresh();
  } catch (error) {
    state.messages.push({ kind: 'error', text: error.message, question: mode === 'ask' ? text : null });
    // Keep draft and File objects on failed writes; retry does not lose user input.
  } finally {
    state.busy = null; updateFeed(); busyControls(); $('#composer-text')?.focus();
  }
}
function openDialog(title, body, { wide = false } = {}) {
  lastDialogTrigger = document.activeElement;
  dialog.classList.toggle('wide', wide);
  $('#dialog-content').innerHTML = demoCopy(`<header class="dialog-header"><h2 id="dialog-title">${esc(title)}</h2><button class="icon-button" data-action="close-dialog" aria-label="关闭面板">${icon('close')}</button></header><div class="dialog-body">${body}</div>`);
  if (!dialog.open) dialog.showModal();
}
function closeDialog() { dialog.close(); lastDialogTrigger?.focus?.(); }
async function confirmAction(title, description, label = '确认', danger = true) {
  const el = $('#confirm-dialog'); $('#confirm-title').textContent = title; $('#confirm-description').textContent = description;
  $('#confirm-accept').textContent = label; $('#confirm-accept').className = `button ${danger ? 'danger' : 'primary'}`;
  el.returnValue = 'cancel'; el.showModal();
  return new Promise(resolve => el.addEventListener('close', () => resolve(el.returnValue === 'accept'), { once: true }));
}
async function recordDetail(id) {
  const result = await api(`/api/workspace/records/${encodeURIComponent(id)}`);
  openDialog('记忆原文与状态轨迹', `<div class="detail-kicker">${icon('timeline')}${date(result.created_at, { year: 'numeric' })} ${badge(result.consolidated === 0 ? '整理中' : '本地记录')}</div><h3 class="detail-section-title">当时，你这样记下</h3><div class="original-text">${esc(result.content || '这条记录没有文字，请查看附件原件。')}</div>${result.documents.length ? `<h3 class="detail-section-title">关联原件 · ${result.documents.length} 份</h3><div class="detail-file-list">${result.documents.map(d => `<div>${icon('file')}<span>${esc(d.title)}</span>${originalLink(d.id)}</div>`).join('')}</div>` : ''}<h3 class="detail-section-title">整理出的记忆 · ${result.facts.length} 个片段</h3>${result.facts.length ? result.facts.map(f => `<article class="fact-detail">${badge(f.valid_to ? '历史状态' : '当前有效', f.valid_to ? 'neutral' : '')}<strong>${esc(f.summary)}</strong><p>${esc(f.fields.place ? '位置：' + f.fields.place : f.fields.datetime_hint ? '时间提示：' + f.fields.datetime_hint : '')}</p><small>记录时间 ${date(f.created_at)} · 生效时间 ${date(f.valid_from)}</small></article>`).join('') : `<p class="muted">${result.consolidated === 0 ? '后台正在整理，稍后重新打开即可查看。' : '未提取到结构化事实，原始记录仍可在本地查看。'}</p>`}${result.history.length ? `<h3 class="detail-section-title">同一事物的状态变化</h3><div class="state-history">${result.history.map(f => `<div><span class="state-dot"></span><small>${date(f.valid_from, { year: 'numeric' })} ${f.valid_to ? '→ ' + date(f.valid_to) : '至今'}</small><p>${esc(f.summary)}</p>${f.raw_id !== id ? button('查看这一版原文', 'record-detail', { id: f.raw_id, cls: 'subtle small', icon: 'external' }) : badge('当前查看的记录', 'neutral')}</div>`).join('')}</div>` : ''}`);
}
async function filePreview(id) {
  const doc = state.documents.find(d => d.id === id);
  if (!doc) throw new Error('原件已移除，请刷新后重试。');
  const result = await api(`/api/files/${encodeURIComponent(id)}`);
  const isPhoto = doc.kind === 'photo', pdf = /\.pdf$/i.test(doc.title || '');
  const type = (doc.title || '').split('.').pop().toUpperCase();
  const typeLabel = type === 'TXT' ? 'TXT · 纯文本' : type === 'MD' ? 'Markdown · 文本' : type || '文件';
  openDialog(doc.title || '原件预览', `<div class="preview-toolbar">${badge(typeLabel, 'neutral')}${badge(doc.allow_cloud ? '已授权附件云分析' : '仅本地处理')}${badge(result.status === 'done' ? `已解析 · ${result.chunk_count} 个片段` : result.status === 'failed' ? '解析失败' : '正在解析', 'neutral')}<span>${bytes(doc.size_bytes)}</span></div>${doc.size_bytes != null && isPhoto ? `<div class="full-image"><img src="/api/files/${encodeURIComponent(id)}/raw" alt="${attr(doc.title)}"></div>` : doc.size_bytes != null && pdf ? `<iframe class="pdf-viewer" title="${attr(doc.title)}" src="/api/files/${encodeURIComponent(id)}/raw"></iframe>` : ''}${result.summary ? `<h3 class="detail-section-title">摘要</h3><p class="original-text">${esc(result.summary)}</p>` : ''}<h3 class="detail-section-title">解析文本预览</h3><p class="original-text">${esc(result.parsed_text || (result.status === 'processing' ? '后台处理中，完成后重新打开即可查看。' : result.status === 'failed' ? '解析暂未成功。可下载原件检查，或重试解析。' : '未识别到可预览的文本，原件仍可下载。'))}</p><p class="muted">显示前 500 字。点击“打开原件”查看文件正文。</p><div class="dialog-actions">${originalLink(id)}${originalLink(id, '下载原件', true)}${doc.raw_id ? button('查看关联记录', 'record-detail', { id: doc.raw_id, cls: 'secondary', icon: 'link' }) : ''}${result.status === 'failed' ? button('重试解析', 'retry-file', { id, cls: 'primary', icon: 'refresh' }) : ''}</div>`, { wide: pdf || isPhoto });
}
async function originalView(id, pageNumber) {
  const doc = state.documents.find(d => d.id === id);
  const title = doc?.title || '原件';
  const url = `/api/files/${encodeURIComponent(id)}/raw?disposition=inline`;
  const blob = await api(url, { blob: true });
  let content;
  if (blob.type.startsWith('text/plain')) {
    const text = await blob.text();
    content = `<p class="original-file-type">纯文本原件 · UTF-8 · ${bytes(blob.size)}</p><pre class="original-text original-full-text">${esc(text.slice(0, 200000))}</pre>${text.length > 200000 ? '<p class="muted">已显示前 20 万字，完整内容请下载原件。</p>' : ''}`;
  } else if (blob.type === 'application/pdf') {
    content = `<iframe class="pdf-viewer" title="${attr(title)}" src="${url}${pageNumber ? '#page=' + Number(pageNumber) : ''}"></iframe>`;
  } else if (/^image\/(png|jpeg|gif|webp|bmp)$/.test(blob.type)) {
    content = `<div class="full-image"><img src="${url}" alt="${attr(title)}"></div>`;
  } else {
    content = `<p class="original-text">${esc(title)} 需要对应的文档应用打开，请下载原件后查看。解析预览仍可在文件库中查看。</p>`;
  }
  openDialog(title, content + `<div class="dialog-actions">${originalLink(id, '下载原件', true)}${button('返回解析预览', 'file-preview', { id, cls: 'secondary' })}</div>`, { wide: true });
}
function privacy() {
  openDialog('你的记忆，如何被保管', `<div class="privacy-heading">${icon('shield')}<h3>保存在本地，权限由你决定。</h3></div><div class="privacy-list"><article><strong>原文与文件，存放在你的本机</strong><p>使用 SQLite 与本地文件目录，不自动同步到其他设备。随时可以导出 Markdown 或下载原件。</p></article><article><strong>“仅本地”指附件权限</strong><p>默认不把图片和文件发送给云模型。勾选“允许附件云分析”，才授权这些附件用于云摘要或图片理解。附件中的隐私内容可能因此发送到模型服务。</p></article><article><strong>文字抽取与问答可能调用已配置的云模型</strong><p>文字在出库前按原项目规则对敏感值打码。打码不能保证识别所有隐私；未授权附件的分析效果也会受到限制。</p></article><article><strong>删除有一次回头的机会</strong><p>回收站保留 30 天。到期清理在服务启动时执行；彻底删除不可撤销，也不等同于磁盘覆写。</p></article><article><strong>当前是本机单用户空间</strong><p>应用没有账号登录。请通过本机地址访问，不向不受信任的网络公开服务。</p></article></div><div class="privacy-status">${icon('check')}本地记录服务已连接 · ${state.overview.cloud_configured ? '已配置云模型' : '未配置云模型密钥'}</div>`);
}
function uploadDialog() {
  uploadFiles = [];
  openDialog('把原件留在记忆里', `<p class="dialog-description">照片、PDF、Word、文本与 Excel，保留原件，也保留可检索的线索。</p><label class="upload-drop" id="upload-drop" tabindex="0">${icon('folder')}<strong>选择文件，或把文件拖到这里</strong><span>可一次选择多份文件 · 原件留存在本地</span><input type="file" id="upload-input" multiple hidden></label><div id="upload-selected"></div><label class="upload-consent"><input type="checkbox" id="upload-cloud">允许本次附件进行云分析</label><p class="privacy-hint">默认仅本地解析。授权后，附件中的内容可能发送到已配置的模型服务，换取摘要和图片理解。</p><div class="dialog-actions">${button('开始上传', 'upload-confirm', { cls: 'primary', icon: 'plus' })}</div>`);
}
function uploadSelection(files) { uploadFiles = [...files]; $('#upload-selected').innerHTML = uploadFiles.map(f => `<div class="upload-file-row">${icon('file')}<span>${esc(f.name)}</span><small>${bytes(f.size)}</small></div>`).join(''); }
async function uploadConfirm(el) {
  if (!uploadFiles.length) { toast('先选择一份文件。'); return; }
  const allowCloud = $('#upload-cloud').checked, files = [...uploadFiles], failed = [];
  el.disabled = true; let succeeded = 0;
  for (const file of files) {
    el.textContent = `正在上传 ${succeeded + failed.length + 1} / ${files.length}`;
    const form = new FormData(); form.append('file', file); form.set('allow_cloud', allowCloud ? '1' : '0');
    try { await post('/api/files', form); succeeded++; } catch (error) { failed.push({ file, message: error.message }); }
  }
  await refresh();
  if (failed.length) {
    uploadFiles = failed.map(r => r.file);
    $('#upload-selected').innerHTML = failed.map(r => `<div class="upload-file-row error">${icon('info')}<span>${esc(r.file.name)}<small>${esc(r.message)}</small></span></div>`).join('');
    el.disabled = false; el.textContent = '重试未成功的文件'; toast(`已保存 ${succeeded} 份，${failed.length} 份未成功。`, true);
  } else { closeDialog(); toast(`已保存 ${succeeded} 份原件，正在后台整理。`); if (state.route !== 'files') location.hash = '#files'; else render(); }
}
async function semanticSearch(el) {
  const question = state.fileSearch.trim(); if (!question) { toast('先输入你要查找的文件内容。'); $('#file-search').focus(); return; }
  el.disabled = true; const requestQuestion = question;
  try {
    const result = await post('/api/documents/search', { question, top_k: 10 });
    if (state.fileSearch !== requestQuestion || !$('#semantic-results')) return;
    $('#semantic-results').innerHTML = `<section class="semantic-results"><div class="semantic-heading"><h2>${icon('spark')}文档内检索结果 · ${result.results.length} 个片段</h2>${button('收起', 'clear-semantic', { cls: 'subtle small', icon: 'close' })}</div>${result.results.length ? sourcePanel({ documents: result.results }) : '<p>没有找到相关片段。可更换关键词或检查文件是否已完成解析。</p>'}</section>`;
  } finally { el.disabled = false; }
}
function findReminder(id) { return [...state.reminders.due, ...state.reminders.pending, ...(state.reminders.completed || [])].find(r => r.fact_id === id); }
function reminderDetail(id) {
  const r = findReminder(id); if (!r) throw new Error('提醒已变更，请刷新后重试。');
  let localValue = '';
  if (r.due_at) { const dt = new Date(r.due_at); localValue = new Date(dt.getTime() - dt.getTimezoneOffset() * 60000).toISOString().slice(0, 16); }
  openDialog('给这件事定个时间', `<p class="reminder-dialog-summary">${esc(r.summary)}</p><label class="form-label" for="reminder-time">提醒时间</label><input type="datetime-local" id="reminder-time" class="text-input" value="${attr(localValue)}"><p class="privacy-hint">按当前设备时区保存，重新打开页面也不会漂移。</p><div class="dialog-actions">${button('保存提醒时间', 'save-reminder-time', { id, cls: 'primary', icon: 'check' })}</div>`);
}
async function loadSpending(period = spendingPeriod) {
  spendingPeriod = period; spendingData = await api(`/api/workspace/spending?period=${encodeURIComponent(period)}`);
  if (!spendingData.totals[spendingCurrency]) spendingCurrency = Object.keys(spendingData.totals)[0];
  renderSpending();
}
function renderSpending() {
  const d = spendingData, rows = d.rows.filter(r => r.currency === spendingCurrency), sum = d.totals[spendingCurrency] || 0;
  const weeks = [0, 0, 0, 0, 0]; rows.forEach(r => { const i = Math.min(4, Math.floor((new Date(r.occurred).getDate() - 1) / 7)); weeks[i] += r.amount; });
  const max = Math.max(...weeks, 1);
  openDialog('消费记忆分析', `<div class="spending-toolbar"><label>统计范围 <select id="spending-period"><option value="this_month" ${spendingPeriod === 'this_month' ? 'selected' : ''}>本月</option><option value="last_month" ${spendingPeriod === 'last_month' ? 'selected' : ''}>上月</option></select></label>${Object.keys(d.totals).length > 1 ? `<label>币种 <select id="spending-currency">${Object.keys(d.totals).map(c => `<option ${c === spendingCurrency ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select></label>` : ''}</div><div class="spending-total"><span>${esc(spendingCurrency || '暂无消费数据')}</span><strong>${sum.toFixed(2)}</strong><p>${rows.length} 笔有金额的消费记录 · 另有 ${d.missing} 笔未识别到金额</p></div><div class="spending-chart" role="img" aria-label="按月内周次汇总的消费柱状图，币种 ${attr(spendingCurrency || '未标币种')}">${weeks.map((v, i) => `<div class="chart-column"><span>${v.toFixed(2)}</span><div class="bar-track"><div class="chart-bar" style="--bar:${Math.max(0, v / max * 100)}%"></div></div><small>第 ${i + 1} 周</small></div>`).join('')}</div><p class="chart-disclosure">统计口径：按记录时间 ${date(d.start, { year: 'numeric', hour: undefined, minute: undefined })} 至 ${date(new Date(new Date(d.end).getTime() - 1), { year: 'numeric', hour: undefined, minute: undefined })}；仅消费类型，按币种分别求和。退款等负值保留在合计与明细中。</p><h3 class="detail-section-title">明细清单</h3>${rows.length ? `<div class="spending-rows">${rows.map(r => `<div><span>${date(r.occurred)}</span><p>${esc(r.summary)}</p><strong>${r.amount.toFixed(2)}</strong>${r.raw_id ? button('原记录', 'record-detail', { id: r.raw_id, cls: 'subtle small', icon: 'external' }) : ''}</div>`).join('')}</div>` : empty('这个月，还没有消费记忆', '记录时写明金额，整理完成后便可在这里查看。', 'coins')}`, { wide: true });
}
async function voice() {
  if (speech && state.listening) { speech.stop(); return; }
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) { toast('当前浏览器不提供语音识别。请使用支持语音识别的 Chrome / Edge，或直接输入文字。', true); return; }
  const consent = await confirmAction('开始浏览器语音口述？', '语音识别由浏览器提供，可能使用浏览器厂商的在线服务。识别后的文字只填入输入框，确认记录前不会保存。请勿口述敏感信息。', '允许并开始', false);
  if (!consent) return;
  speech = new Recognition(); speech.lang = 'zh-CN'; speech.interimResults = true; speech.continuous = true;
  const base = state.draft;
  speech.onstart = () => { state.listening = true; updateVoiceButton(); toast('正在聆听，点击“结束口述”停止。'); };
  speech.onresult = e => { let text = ''; for (const r of e.results) text += r[0].transcript; state.draft = base + (base ? ' ' : '') + text; if ($('#composer-text')) $('#composer-text').value = state.draft; };
  speech.onerror = e => toast(e.error === 'not-allowed' ? '麦克风权限未开启，可在浏览器中授权。' : '语音识别暂不可用，请稍后重试或输入文字。', true);
  speech.onend = () => { state.listening = false; speech = null; updateVoiceButton(); };
  try { speech.start(); } catch { state.listening = false; speech = null; toast('语音识别无法启动，请输入文字。', true); }
}
function updateVoiceButton() { const el = $('[data-action="voice"]'); if (el) { el.classList.toggle('listening', state.listening); el.innerHTML = icon('mic') + `<span>${state.listening ? '结束口述' : '语音口述'}</span>`; } }

document.addEventListener('click', async event => {
  const el = event.target.closest('[data-action]'); if (!el || el.disabled) return;
  const action = el.dataset.action, id = el.dataset.id;
  if (el.tagName === 'A' && !['focus-composer', 'original-view'].includes(action)) return;
  event.preventDefault();
  if (el.dataset.pending) return;
  el.dataset.pending = 'true';
  try {
    switch (action) {
      case 'menu': { const open = document.body.classList.toggle('menu-open'); el.setAttribute('aria-expanded', String(open)); break; }
      case 'close-menu': closeMenu(); break;
      case 'refresh': await refresh({ manual: true }); break;
      case 'privacy': privacy(); break;
      case 'model-settings': await modelSettings(openDialog, toast, refresh); break;
      case 'original-view': await originalView(id, el.dataset.page); break;
      case 'close-dialog': closeDialog(); break;
      case 'go-chat': case 'focus-composer': goChat(); break;
      case 'prompt': goChat(el.dataset.text); break;
      case 'supplement': closeDialog(); goChat(''); toast(`补充一条与“${el.dataset.text}”相关的真实记录。`); break;
      case 'new-reminder': openReminderCreate(); break;
      case 'create-reminder': await createReminder(el); break;
      case 'toggle-layout': state.chatLayout = state.chatLayout === 'split' ? 'single' : 'split'; render(); break;
      case 'clear-chat': if (await confirmAction('清空当前对话视图？', '只清除当前页面的对话与追问上下文，已保存的记忆和提问历史仍保留。', '清空视图', false)) { state.messages = []; state.qaLog = []; updateFeed(); } break;
      case 'record': await submit('record'); break;
      case 'ask': await submit('ask'); break;
      case 'reask': closeDialog(); goChat(el.dataset.text); await submit('ask', el.dataset.text); break;
      case 'attach': $('#attachment-input')?.click(); break;
      case 'remove-attachment': state.attachments.splice(Number(id), 1); $('#attachment-list').innerHTML = views.attachments(state); break;
      case 'voice': await voice(); break;
      case 'record-detail': await recordDetail(id); break;
      case 'file-preview': await filePreview(id); break;
      case 'file-filter': state.fileFilter = id; render(); break;
      case 'timeline-filter': state.timelineFilter = id; render(); break;
      case 'reminder-filter': state.reminderFilter = id; render(); break;
      case 'trash-filter': state.trashFilter = id; render(); break;
      case 'history-period': state.historyPeriod = id; render(); await loadHistory(); break;
      case 'more-history': await loadHistory(true); break;
      case 'more-memories': state.memoryLimit += 50; await refresh(); break;
      case 'history-detail': { const row = await api(`/api/workspace/history/${encodeURIComponent(id)}`); openDialog('回看答案与证据来源', `<div class="detail-question">${esc(row.question)}</div>${answerCard(row, row.question, { history: true })}<p class="privacy-hint">历史答案按当时内容留存。已删除的原件会停止提供预览，历史引用文字不随之改写。</p>`); break; }
      case 'clear-history': if (await confirmAction('清空全部提问历史？', '将删除全部问答历史及其答案缓存，无法恢复。记忆与原件会保留。', '清空历史')) { await api('/api/qa-history', { method: 'DELETE' }); state.history = { items: [], total: 0 }; await refresh(); render(); toast('提问历史已清空'); } break;
      case 'upload': uploadDialog(); break;
      case 'upload-confirm': await uploadConfirm(el); break;
      case 'semantic-search': await semanticSearch(el); break;
      case 'clear-semantic': $('#semantic-results').innerHTML = ''; break;
      case 'retry-file': await post(`/api/workspace/files/${encodeURIComponent(id)}/retry`, {}); closeDialog(); await refresh(); render(); toast('已重新提交解析'); break;
      case 'delete-record': case 'delete-file': {
        const record = action === 'delete-record';
        if (await confirmAction('把这组记忆暂时放下？', '该记录与同次上传的原件会一起移入回收站，30 天内可恢复。', '移入回收站', false)) {
          await api(`/api/${record ? 'memories' : 'documents'}/${encodeURIComponent(id)}`, { method: 'DELETE' });
          closeDialog(); await refresh(); if (state.route !== 'chat') render(); toast('已移入回收站，可在 30 天内恢复');
        } break;
      }
      case 'restore': { const result = await post(`/api/workspace/trash/${el.dataset.kind}/${encodeURIComponent(id)}/restore-group`, {}); await refresh(); render(); toast(`已恢复 ${result.restored} 个关联条目`); break; }
      case 'purge': if (await confirmAction('彻底删除这组条目？', '该条目及同组仍在回收站的记录、原件与派生索引将永久删除，无法撤销。', '彻底删除')) { await api(`/api/trash/${el.dataset.kind}/${encodeURIComponent(id)}`, { method: 'DELETE' }); await refresh(); render(); toast('已彻底删除'); } break;
      case 'empty-trash': if (await confirmAction('清空回收站？', '站内全部记录与原件将永久删除。', '继续清空') && await confirmAction('最后确认：这些记忆将无法恢复', `当前回收站共有 ${state.trash.records.length + state.trash.documents.length} 个条目，请确认不再需要。`, '永久清空')) { const r = await api('/api/trash', { method: 'DELETE' }); await refresh(); render(); const left = state.trash.records.length + state.trash.documents.length; toast(`已删除 ${r.records} 条记录、${r.documents} 份原件${left ? `，仍有 ${left} 项未成功，请重试` : ''}`, !!left); } break;
      case 'toggle-reminder': { const r = findReminder(id); await api(`/api/reminders/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify({ completed: !r.completed_at }) }); await refresh(); render(); toast(r.completed_at ? '已重新打开提醒' : '这件事，完成了'); break; }
      case 'reminder-detail': reminderDetail(id); break;
      case 'save-reminder-time': { const value = $('#reminder-time').value; if (!value) { toast('请选择提醒时间。'); break; } await api(`/api/reminders/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify({ due_at: new Date(value).toISOString() }) }); closeDialog(); await refresh(); if (state.route !== 'chat') render(); toast('提醒时间已保存'); break; }
      case 'export-markdown': { const response = await fetch('/api/export'); if (!response.ok) throw new Error('导出失败，请稍后重试。'); downloadBlob(await response.blob(), `记忆导出-${new Date().toLocaleDateString('sv-SE')}.md`); toast('Markdown 已生成'); break; }
      case 'export-files': if (!state.selected.size) toast('请先勾选需要导出的文件。'); else { const blob = await post('/api/workspace/files/export', { document_ids: [...state.selected] }, { blob: true }); downloadBlob(blob, '记忆原件.zip'); toast('所选原件已打包'); } break;
      case 'spending': await loadSpending(); break;
    }
  } catch (error) { showError(error); }
  finally { delete el.dataset.pending; }
});

document.addEventListener('input', event => {
  const el = event.target;
  if (el.id === 'composer-text') state.draft = el.value;
  if (el.id === 'file-search') { state.fileSearch = el.value; $('#file-grid').innerHTML = views.fileGrid(state); $('#select-all-files').checked = false; $('#semantic-results').innerHTML = ''; }
  if (el.id === 'timeline-search') { state.timelineSearch = el.value; $('#timeline-results').innerHTML = views.timelineRows(state); }
  if (el.id === 'history-search') { state.historySearch = el.value; clearTimeout(searchTimer); searchTimer = setTimeout(() => loadHistory(), 250); }
});
document.addEventListener('change', async event => {
  const el = event.target;
  try {
    if (el.id === 'attachment-input') { state.attachments.push(...el.files); el.value = ''; $('#attachment-list').innerHTML = views.attachments(state); }
    if (el.id === 'allow-cloud') state.allowCloud = el.checked;
    if (el.id === 'upload-input') uploadSelection(el.files);
    if (el.id === 'cloud-filter') { state.cloudFilter = el.value; $('#file-grid').innerHTML = views.fileGrid(state); }
    if (el.dataset.fileId) { el.checked ? state.selected.add(el.dataset.fileId) : state.selected.delete(el.dataset.fileId); $('#selected-count').textContent = `已选 ${state.selected.size} 份`; }
    if (el.id === 'select-all-files') { for (const d of views.filteredFiles(state)) el.checked ? state.selected.add(d.id) : state.selected.delete(d.id); $('#file-grid').innerHTML = views.fileGrid(state); $('#selected-count').textContent = `已选 ${state.selected.size} 份`; }
    if (el.id === 'spending-period') await loadSpending(el.value);
    if (el.id === 'spending-currency') { spendingCurrency = el.value; renderSpending(); }
  } catch (error) { showError(error); }
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') closeMenu();
  if (event.isComposing) return;
  if (event.target.id === 'composer-text' && event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submit(event.ctrlKey || event.metaKey ? 'ask' : 'record'); }
  if (event.target.id === 'file-search' && event.key === 'Enter') { event.preventDefault(); semanticSearch($('[data-action="semantic-search"]')).catch(showError); }
  if (event.key === '/' && !event.target.matches('input,textarea,select') && !dialog.open) { event.preventDefault(); $('input[type="search"]')?.focus(); }
  if (event.target.id === 'upload-drop' && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); $('#upload-input').click(); }
});
document.addEventListener('dragover', e => { if (e.target.closest('#upload-drop')) { e.preventDefault(); $('#upload-drop').classList.add('dragging'); } });
document.addEventListener('dragleave', e => e.target.closest('#upload-drop')?.classList.remove('dragging'));
document.addEventListener('drop', e => { if (e.target.closest('#upload-drop')) { e.preventDefault(); $('#upload-drop').classList.remove('dragging'); uploadSelection(e.dataTransfer.files); } });
dialog.addEventListener('click', event => { if (event.target === dialog) { const rect = dialog.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) closeDialog(); } });
window.addEventListener('hashchange', async () => { speech?.stop(); state.route = location.hash.slice(1) || 'chat'; render(); if (state.route === 'history') await loadHistory(); $('#main').scrollTop = 0; });
window.addEventListener('beforeunload', event => { if (state.draft.trim() || state.attachments.length || state.busy) { event.preventDefault(); event.returnValue = ''; } });
document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });

function openReminderCreate() {
  openDialog('留下一件要惦记的事', `<label class="form-label" for="new-reminder-text">提醒内容</label><textarea id="new-reminder-text" class="text-input" rows="3" placeholder="例如：交房租、买牛奶、带上体检报告"></textarea><label class="form-label" for="new-reminder-time">提醒时间（可稍后设定）</label><input type="datetime-local" id="new-reminder-time" class="text-input"><p class="privacy-hint">直接保存到本地提醒，不需要等待 AI 识别。打开页面时即可查看。</p><div class="dialog-actions">${button('保存提醒', 'create-reminder', { cls: 'primary', icon: 'check' })}</div>`);
}
async function createReminder(el) {
  const summary = $('#new-reminder-text').value.trim(), time = $('#new-reminder-time').value;
  if (!summary) { toast('先写下需要提醒的事。'); $('#new-reminder-text').focus(); return; }
  el.disabled = true;
  try { await post('/api/reminders', { summary, due_at: time ? new Date(time).toISOString() : null }); closeDialog(); await refresh(); if (state.route !== 'reminders') location.hash = '#reminders'; else render(); toast('提醒已保存在本地'); }
  finally { el.disabled = false; }
}

hydrateIcons(); render();
await refresh();
if (state.route === 'history') await loadHistory();
setInterval(() => { if (!document.hidden) refresh(); }, 10000);
