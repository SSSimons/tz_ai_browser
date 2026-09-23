"""Общие проверки видимых слоёв страницы без правил отдельных сайтов."""

# Один запрос к DOM вместо последовательных обращений к каждому кандидату.
# Порядок HTML не определяет порядок наложения окон: проверяем попадание указателя.
FIND_MODAL = """token => {
    const visible = e => {
        const s = getComputedStyle(e), r = e.getBoundingClientRect();
        return e.isConnected && !e.closest('[inert], [aria-hidden="true"]') &&
            s.visibility !== 'hidden' && s.display !== 'none' && Number(s.opacity) > 0 &&
            r.width > 0 && r.height > 0;
    };
    const controls = 'button, input, textarea, select, [contenteditable=true], [role=button]';
    const semantic = [...document.querySelectorAll(
        'dialog[open], [role=dialog], [role=alertdialog], [aria-modal=true]'
    )].filter(e => visible(e) && e.getAttribute('aria-modal') !== 'false');
    // Без ARIA распознаём только перекрывающий почти весь экран слой с контролами.
    const layers = [...document.querySelectorAll('body *')].filter(e => {
        const s = getComputedStyle(e);
        if (s.position !== 'fixed' || s.pointerEvents === 'none' || !visible(e)) return false;
        const r = e.getBoundingClientRect();
        return r.width >= innerWidth * .8 && r.height >= innerHeight * .8 &&
            !!e.querySelector(controls);
    });
    const candidates = [...new Set([...layers, ...semantic])];
    if (!candidates.length) return null;
    const exposed = e => {
        const nodes = [e, ...e.querySelectorAll(controls)].slice(0, 25);
        return nodes.some(n => {
            const r = n.getBoundingClientRect();
            const x = Math.max(0, Math.min(innerWidth - 1, r.x + r.width / 2));
            const y = Math.max(0, Math.min(innerHeight - 1, r.y + r.height / 2));
            const hit = document.elementFromPoint(x, y);
            return hit && (hit === e || e.contains(hit));
        });
    };
    const exposedCandidates = candidates.filter(exposed);
    const available = exposedCandidates.length ? exposedCandidates : candidates;
    // Вложенное окно приоритетнее охватывающего его backdrop.
    const inner = available.filter(e => !available.some(n => n !== e && e.contains(n)));
    let top = inner[inner.length - 1];
    for (const e of inner) {
        if (e.matches(':modal') && !top.matches(':modal')) { top = e; continue; }
        const a = e.getBoundingClientRect(), b = top.getBoundingClientRect();
        const left = Math.max(a.left, b.left, 0), right = Math.min(a.right,b.right,innerWidth);
        const up = Math.max(a.top,b.top,0), down = Math.min(a.bottom,b.bottom,innerHeight);
        if (left >= right || up >= down) continue;
        const stack = document.elementsFromPoint((left+right)/2, (up+down)/2);
        const ei = stack.findIndex(n => n === e || e.contains(n));
        const ti = stack.findIndex(n => n === top || top.contains(n));
        if (ei >= 0 && (ti < 0 || ei < ti)) top = e;
    }
    if (!top.hasAttribute('data-browser-agent-scope')) {
        top.setAttribute('data-browser-agent-scope', token);
    }
    return top.getAttribute('data-browser-agent-scope');
}"""

# Состояние для защиты от повторной отправки без изменения страницы или полей.
# Значения остаются только в памяти процесса и не передаются в журнал.
CLICK_STATE = """e => {
    const root = e.closest('[data-browser-agent-scope]') || document.body;
    return JSON.stringify([location.href, root.innerText, e.outerHTML,
        [...root.querySelectorAll('input, textarea, select, [contenteditable=true]')]
            .map(n => [n.value, n.checked, n.isContentEditable ? n.innerText : null])]);
}"""
