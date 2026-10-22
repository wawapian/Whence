import { api } from './api.js';
import { esc, attr } from './components.js';

const field = (id, label, value = '', options = '') => `<label class="settings-field" for="${id}"><span>${label}</span><input id="${id}" class="text-input" value="${attr(value ?? '')}" ${options}></label>`;
const keyField = (prefix, configured) => `<label class="settings-field" for="${prefix}-key"><span>API Key <small>${configured ? '已配置' : '未配置'}</small></span><input id="${prefix}-key" class="text-input" type="password" autocomplete="new-password" maxlength="500" placeholder="${configured ? '留空保留当前 Key，填写新 Key 即可替换' : '填写你的 API Key'}"></label><p class="settings-note">填写新 Key 后保存即可替换；同一服务地址下留空则保留已保存的 Key。</p>`;

export async function modelSettings(openDialog, toast, refresh) {
  const data = await api('/api/model-settings');
  const presets = data.providers;
  const names = Object.fromEntries(Object.entries(presets).map(([id, item]) => [id, item.name]));
  openDialog('模型设置', `<form id="model-settings-form" autocomplete="off">
    <p class="settings-intro">${'保存在本机，重启后仍然有效。'}保存后，新的提问和记录会使用这份配置。</p>
    <fieldset><legend>文字模型</legend><label class="settings-field" for="model-provider"><span>服务商</span><select id="model-provider" class="text-input">${Object.entries(names).map(([id, label]) => `<option value="${id}" ${data.llm_provider === id || (id === 'custom' && !names[data.llm_provider]) ? 'selected' : ''}>${label}</option>`).join('')}</select></label>
    ${field('llm-url', 'Base URL', data.llm_base_url, 'type="url" required maxlength="500"')}
    ${field('llm-model', '模型 ID（可选择建议，也可手填）', data.llm_model, 'required maxlength="160" list="model-suggestions" placeholder="填写服务商的 API 模型 ID"')}
    <datalist id="model-suggestions"></datalist><button type="button" class="button subtle" id="model-use-preset">使用当前建议模型</button>
    <p class="settings-note" id="model-provider-note"></p>
    ${keyField('llm', data.llm_key_configured)}
    <p class="settings-note">所有列出的服务商都有 API，需要各自平台的 API Key。Key 加密保存在当前空间，不会回显。更换服务地址时，请重新填写 Key。模型是否可调用还取决于账户权限与余额。</p></fieldset>
    <details class="settings-details"><summary>生成参数</summary><div class="settings-grid">
    ${field('model-temperature', 'Temperature · 随机程度', data.llm_temperature, 'type="number" min="0" max="2" step="0.1" placeholder="使用服务商默认值"')}
    ${field('model-tokens', '每次调用的输出 Token 上限', data.llm_max_tokens, 'type="number" min="128" max="16384" step="1" placeholder="使用服务商默认值"')}
    ${field('model-timeout', '网络超时（秒）', data.llm_timeout_s, 'type="number" min="1" max="120" required')}
    ${field('model-rounds', '最多工具调用轮数', data.qa_max_rounds, 'type="number" min="1" max="8" required')}
    </div><p id="model-parameter-note" class="settings-note" role="status"></p>
    <p class="settings-note">输出上限限制单次生成，不保证达到该长度；思考模型会把推理 Token 计入上限。工具轮数控制检索次数。图片模型仅共享输出上限和超时，文字温度不应用到图片模型。</p>
</details>
    <details class="settings-details"><summary>图片理解模型（可选）</summary>
    ${field('vlm-url', 'Base URL', data.vlm_base_url, 'type="url" required maxlength="500"')}
    ${field('vlm-model', '图片模型名称', data.vlm_model, 'maxlength="160" placeholder="留空关闭图片理解"')}
    ${keyField('vlm', data.vlm_key_configured)}</details>
    <p id="model-settings-message" class="settings-message" role="status" aria-live="polite"></p>
    <div class="dialog-actions settings-actions"><button type="button" class="button subtle" id="model-settings-reset">恢复默认配置</button><button type="submit" class="button primary" id="model-settings-save">保存设置</button></div>
    </form>`, { wide: false });
  const form = document.querySelector('#model-settings-form');
  const dialog = document.querySelector('#detail-dialog');
  const get = id => form.querySelector('#' + id);
  const message = get('model-settings-message');
  let rules = data.parameter_policy;
  let revision = 0;
  let pending = false;
  const showRules = () => {
    const temperature = get('model-temperature');
    temperature.disabled = !rules.temperature_supported;
    temperature.max = rules.temperature_max;
    if (temperature.disabled) temperature.value = '';
    temperature.placeholder = temperature.disabled ? '服务商固定值，不发送此参数' : '使用服务商默认值';
    get('model-parameter-note').textContent = rules.note || '温度和输出上限将发送给服务商；自定义模型需自行核对参数支持情况。';
    get('model-settings-save').disabled = pending || rules.blocked;
  };
  const showProvider = () => {
    const preset = presets[get('model-provider').value];
    get('model-suggestions').innerHTML = preset.models.map(model => `<option value="${attr(model)}"></option>`).join('');
    get('model-provider-note').innerHTML = `${esc(preset.note)} ${preset.docs ? `<a href="${attr(preset.docs)}" target="_blank" rel="noopener noreferrer">官方文档</a>` : ''}`;
  };
  const showKeyStatus = () => {
    const changed = get('llm-url').value.trim().replace(/\/$/, '') !== data.llm_base_url.replace(/\/$/, '');
    const configured = data.llm_key_configured && !changed;
    form.querySelector('label[for="llm-key"] small').textContent = configured ? '已配置' : '需填写当前服务的 Key';
    get('llm-key').placeholder = configured ? '留空保留当前 Key，填写新 Key 即可替换' : '填写当前服务商的 API Key';
  };
  const updateRules = async () => {
    showKeyStatus();
    const current = ++revision;
    pending = true; get('model-settings-save').disabled = true;
    try {
      const next = await api('/api/model-settings/policy', { method: 'POST', body: JSON.stringify({ llm_model: get('llm-model').value.trim() || undefined, llm_base_url: get('llm-url').value.trim() || undefined }) });
      if (current !== revision) return;
      rules = next; pending = false; message.textContent = ''; showRules();
    } catch (error) { if (current === revision) { message.textContent = error.message; } }
  };
  showProvider(); showRules();
  let policyTimer;
  for (const id of ['llm-model', 'llm-url']) get(id).addEventListener('input', () => {
    revision++;
    pending = true; get('model-settings-save').disabled = true;
    clearTimeout(policyTimer); policyTimer = setTimeout(updateRules, 250);
  });
  get('model-use-preset').addEventListener('click', () => {
    const preset = presets[get('model-provider').value];
    if (preset.models.length) get('llm-model').value = preset.models[0];
    updateRules();
  });
  dialog.addEventListener('close', () => form.querySelectorAll('input[type=password]').forEach(input => { input.value = ''; }), { once: true });
  get('model-provider').addEventListener('change', event => {
    const preset = presets[event.target.value];
    if (preset.base_url) { get('llm-url').value = preset.base_url; get('llm-model').value = preset.models[0]; }
    get('llm-key').value = '';
    showProvider(); updateRules();
  });
  form.addEventListener('submit', async event => {
    event.preventDefault();
    const submit = get('model-settings-save');
    if (submit.disabled || pending || rules.blocked) return;
    submit.disabled = true; message.textContent = '正在保存…';
    const number = id => get(id).value === '' ? null : Number(get(id).value);
    const body = {
      llm_provider: get('model-provider').value, llm_base_url: get('llm-url').value.trim(),
      llm_model: get('llm-model').value.trim(), llm_api_key: get('llm-key').value,
      llm_temperature: number('model-temperature'), llm_max_tokens: number('model-tokens'),
      llm_timeout_s: number('model-timeout'), qa_max_rounds: number('model-rounds'),
      vlm_base_url: get('vlm-url').value.trim(), vlm_model: get('vlm-model').value.trim(),
      vlm_api_key: get('vlm-key').value,
    };
    try {
      await api('/api/model-settings', { method: 'PUT', body: JSON.stringify(body) });
      get('llm-key').value = ''; get('vlm-key').value = '';
      dialog.close(); toast('模型设置已保存，下次提问起生效'); await refresh({ repaint: false });
    } catch (error) { message.textContent = error.message; }
    finally { submit.disabled = pending || rules.blocked; }
  });
  get('model-settings-reset').addEventListener('click', async event => {
    event.currentTarget.disabled = true;
    try {
      await api('/api/model-settings', { method: 'DELETE' });
      toast('已恢复默认模型配置'); await modelSettings(openDialog, toast, refresh);
    } catch (error) { message.textContent = error.message; event.currentTarget.disabled = false; }
  });
}
