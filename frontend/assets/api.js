export class ApiError extends Error { constructor(message, status) { super(message); this.status = status; } }
export async function api(path, options = {}) {
  let response;
  try { response = await fetch(path, { ...options, headers: { ...(options.body && !(options.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}), ...options.headers } }); }
  catch { throw new ApiError('无法连接本地服务。请确认后端已启动，再刷新重试。', 0); }
  if (!response.ok) {
    let body; try { body = await response.json(); } catch { /* Non-JSON service failure. */ }
    const detail = body?.detail;
    throw new ApiError(typeof detail === 'string' ? detail : Array.isArray(detail) ? '输入信息不完整或格式不正确，请检查后重试。' : `请求失败（${response.status}），请稍后重试。`, response.status);
  }
  if (response.status === 204) return null;
  if (options.blob) return response.blob();
  return response.json();
}
export const post = (path, body, options = {}) => api(path, { method: 'POST', body: body instanceof FormData ? body : JSON.stringify(body), ...options });
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob), a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
