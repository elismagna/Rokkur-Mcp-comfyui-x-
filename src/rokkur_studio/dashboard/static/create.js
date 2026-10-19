// The Create page: tabs, still/clip switch, the prompt the model reads, quick ratings, a
// grid filter, and a quiet refresh of the takes while any is rendering. The form works
// without it (it posts every field); this only hides what does not apply.
(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const form = $('#cr-form');
  if (!form) return;
  const phrases = JSON.parse($('#cr-phrases').textContent || '{}');
  const tabInput = $('#cr-tab'), outInput = $('#cr-output'), prompt = $('#cr-prompt');

  const isClip = () => tabInput.value === 'video' || (tabInput.value === 'cinema' && outInput.value === 'clip');

  const sync = () => {
    const tab = tabInput.value, clip = isClip();
    form.dataset.tab = tab;
    $$('.cr-tab').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === tab)));
    $$('.cr-only').forEach((el) => {
      const show = el.dataset.show === 'clip' ? clip : el.dataset.show === tab;
      el.hidden = !show;
    });
    $$('[data-clip-only]').forEach((el) => { el.hidden = !clip; });
    // only one set of shapes posts: still sizes or clip sizes
    $$('.cr-chips[data-for]').forEach((group) => {
      const on = (group.dataset.for === 'clip') === clip;
      group.hidden = !on;
      $$('input', group).forEach((i) => { i.disabled = !on; });
      if (on && !$('input:checked', group)) $('input', group).checked = true;
    });
    $('#cr-go span').textContent = clip ? 'Make a clip' : 'Make a picture';
    preview();
  };

  const preview = () => {
    const words = prompt.value.trim().replace(/\s+/g, ' ').replace(/\.$/, '');
    const pick = (k) => {
      if (tabInput.value !== 'cinema') return '';
      if (k === 'move' && !isClip()) return '';
      const v = $(`#cam-${k}`)?.value;
      return v ? phrases[k][v] : '';
    };
    const front = ['shot', 'angle', 'move'].map(pick).filter(Boolean);
    const back = ['camera', 'lens', 'focal', 'aperture', 'light'].map(pick).filter(Boolean);
    const parts = [front.join(', '), words, back.join(', ')].filter(Boolean);
    $('#cr-final').textContent = words ? parts.join('. ') : 'Your prompt, as written.';
  };

  $$('.cr-tab').forEach((b) => b.addEventListener('click', () => {
    tabInput.value = b.dataset.tab;
    history.replaceState(null, '', `/ui/create?tab=${b.dataset.tab}`);
    sync();
  }));
  $$('[data-output]').forEach((b) => b.addEventListener('click', () => {
    outInput.value = b.dataset.output;
    $$('[data-output]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    sync();
  }));
  prompt.addEventListener('input', preview);
  $$('.cr-cinema select').forEach((s) => s.addEventListener('change', preview));
  const upload = $('#cr-start-file');
  upload?.addEventListener('change', () => {
    if (upload.files[0]) { const none = $('input[name=start_take][value=""]'); if (none) none.checked = true; }
  });
  form.addEventListener('submit', () => { const go = $('#cr-go'); go.disabled = true; go.querySelector('span').textContent = 'Queuing…'; });

  // a picture chosen to animate opens the Video tab with it selected
  if ($('input[name=start_take]:checked')?.value && tabInput.value === 'image') tabInput.value = 'video';
  sync();

  // -- the takes -----------------------------------------------------------------------------
  const grid = $('#cr-takes');
  let filter = 'all';
  const applyFilter = () => $$('.cr-take', grid).forEach((t) => {
    t.hidden = filter === 'all' ? false
      : filter === 'liked' ? Number(t.dataset.rating) <= 0
      : t.dataset.kind !== filter;
  });
  $$('.cr-filter a').forEach((a) => a.addEventListener('click', (e) => {
    e.preventDefault();
    filter = a.dataset.filter;
    $$('.cr-filter a').forEach((x) => x.classList.toggle('on', x === a));
    applyFilter();
  }));

  const wire = () => {
    $$('form[data-confirm]', grid).forEach((f) => f.addEventListener('submit', (e) => {
      if (!confirm(f.dataset.confirm)) e.preventDefault();
    }));
    $$('.cr-rate', grid).forEach((b) => b.addEventListener('click', async () => {
      const body = new URLSearchParams({ value: b.dataset.value });
      const r = await fetch(`/ui/create/takes/${b.dataset.take}/rate`, { method: 'POST', body });
      if (r.ok) refresh(true);
    }));
    $$('video', grid).forEach((v) => {
      v.addEventListener('mouseenter', () => { v.play().catch(() => {}); });
      v.addEventListener('mouseleave', () => { v.pause(); });
    });
  };

  // Swap in the grid from a fresh copy of the page: the form keeps what you typed.
  const refresh = async (force = false) => {
    if (!force && (document.hidden || grid.dataset.busy !== 'true')) return;
    try {
      const r = await fetch(location.pathname + location.search, { headers: { Accept: 'text/html' } });
      if (!r.ok) return;
      const doc = new DOMParser().parseFromString(await r.text(), 'text/html');
      const fresh = doc.getElementById('cr-takes');
      if (!fresh) return;
      const playing = $$('video', grid).some((v) => !v.paused);
      if (playing && !force) return;
      grid.innerHTML = fresh.innerHTML;
      grid.dataset.busy = fresh.dataset.busy;
      wire(); applyFilter();
    } catch { /* offline: try again next time */ }
  };
  wire();
  setInterval(refresh, 5000);
})();
