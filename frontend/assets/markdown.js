// Small, deliberately text-only Markdown renderer. Escape first; never execute HTML/URLs.
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function inline(text) {
  return escape(text).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
}
const cells = line => line.trim().replace(/^\||\|$/g, '').split('|').map(cell => cell.trim());
export function markdown(value) {
  const lines = String(value || '').replace(/\r\n/g, '\n').split('\n');
  const blocks = []; let paragraph = [], list = [], listType = '', fenced = null;
  function flush() {
    if (paragraph.length) { blocks.push(`<p>${paragraph.map(inline).join('<br>')}</p>`); paragraph = []; }
    if (list.length) { blocks.push(`<${listType}>${list.map(text => `<li>${inline(text)}</li>`).join('')}</${listType}>`); list = []; listType = ''; }
  }
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^\s*```/.test(line)) { flush(); if (fenced !== null) { blocks.push(`<pre><code>${escape(fenced.join('\n'))}</code></pre>`); fenced = null; } else fenced = []; continue; }
    if (fenced !== null) { fenced.push(line); continue; }
    if (!line.trim()) { flush(); continue; }
    if (i + 1 < lines.length && line.includes('|') && /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[i + 1])) {
      flush(); const header = cells(line), rows = []; i += 2;
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) { rows.push(cells(lines[i])); i++; } i--;
      blocks.push(`<div class="markdown-table"><table><thead><tr>${header.map(c => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${header.map((_, col) => `<td>${inline(row[col] || '')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`); continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) { flush(); const level = Math.min(5, heading[1].length + 1); blocks.push(`<h${level}>${inline(heading[2])}</h${level}>`); continue; }
    const item = line.match(/^\s*(?:([-*])|\d+[.)、])\s+(.+)$/);
    if (item) { const type = item[1] ? 'ul' : 'ol'; if (paragraph.length || (listType && listType !== type)) flush(); listType = type; list.push(item[2]); continue; }
    if (list.length) flush();
    paragraph.push(line);
  }
  flush(); if (fenced !== null) blocks.push(`<pre><code>${escape(fenced.join('\n'))}</code></pre>`);
  return blocks.join('');
}
