// Rökkur Studio dashboard behaviour. Everything works without it; this adds instant ratings,
// synced source/render comparison, the live status in the rail, and safe auto-refresh.
(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const store = {
    get: (k, s = localStorage) => { try { return s.getItem(k); } catch { return null; } },
    set: (k, v, s = localStorage) => { try { s.setItem(k, v); } catch { /* private mode */ } },
    del: (k, s = localStorage) => { try { s.removeItem(k); } catch { /* private mode */ } },
  };

  // -- appearance --------------------------------------------------------------------------
  const toggle = $('#theme-toggle');
  const applyTheme = value => {
    document.documentElement.dataset.theme = value;
    if (toggle) toggle.textContent = value === 'light' ? 'Dark appearance' : 'Light appearance';
  };
  applyTheme(store.get('rokkur-theme') || 'dark');
  toggle?.addEventListener('click', () => {
    const value = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    applyTheme(value); store.set('rokkur-theme', value);
  });

  // -- forms: unsaved input pauses refresh; a POST form submits once --------------------------
  let dirty = false;
  document.addEventListener('input', e => { if (e.target.form && !e.target.form.matches('[data-rate]')) dirty = true; });
  document.addEventListener('submit', e => {
    const form = e.target;
    if (e.defaultPrevented || form.method.toLowerCase() !== 'post') return;
    if (form.dataset.submitting) { e.preventDefault(); return; }
    form.dataset.submitting = '1';
    setTimeout(() => { if (e.submitter) { e.submitter.dataset.label = e.submitter.innerHTML; e.submitter.disabled = true; e.submitter.textContent = 'Working…'; } }, 0);
  });
  addEventListener('pageshow', e => { if (!e.persisted) return;
    $$('form[data-submitting]').forEach(form => delete form.dataset.submitting);
    $$('[data-label]').forEach(b => { b.disabled = false; b.innerHTML = b.dataset.label; delete b.dataset.label; });
  });

  // -- New video: keep the draft through a validation error ---------------------------------
  const create = $('#create-form'), draftKey = 'rokkur-create-draft';
  if (create) {
    const fields = [...create.elements].filter(el => el.name && !['file', 'submit', 'hidden'].includes(el.type));
    if (new URLSearchParams(location.search).has('err')) {
      try {
        const draft = JSON.parse(store.get(draftKey, sessionStorage) || '{}');
        fields.forEach(el => { if (!(el.name in draft)) return;
          if (el.type === 'checkbox') el.checked = draft[el.name]; else el.value = draft[el.name];
          el.dispatchEvent(new Event('input')); el.dispatchEvent(new Event('change'));
        });
        if (Object.keys(draft).length) $('#source-info').textContent = 'Your text and settings have been restored. Reselect any uploaded files.';
      } catch { /* a broken draft is ignored */ }
    }
    const saveDraft = () => { const data = {};
      fields.forEach(el => { data[el.name] = el.type === 'checkbox' ? el.checked : el.value; });
      store.set(draftKey, JSON.stringify(data), sessionStorage);
    };
    ['input', 'change', 'submit'].forEach(t => create.addEventListener(t, saveDraft));
  } else if (location.pathname.startsWith('/ui/projects/proj_')) {
    store.del(draftKey, sessionStorage);
  }

  // Taste suggestions on New video: applied only when you press Apply.
  $$('[data-apply]').forEach(btn => btn.addEventListener('click', () => {
    const field = document.getElementsByName(btn.dataset.field)[0];
    if (!field) return;
    if (btn.dataset.action === 'append') {
      const parts = field.value.split(',').map(s => s.trim()).filter(Boolean);
      if (!parts.map(s => s.toLowerCase()).includes(btn.dataset.value.toLowerCase())) parts.push(btn.dataset.value);
      field.value = parts.join(', ');
      field.closest('details')?.setAttribute('open', '');
    } else {
      field.value = btn.dataset.value;
    }
    ['input', 'change'].forEach(t => field.dispatchEvent(new Event(t, { bubbles: true })));
    btn.textContent = 'Applied'; btn.disabled = true;
  }));

  // -- attempt picker -----------------------------------------------------------------------
  $$('[data-preview-target]').forEach(select => select.addEventListener('change', () => {
    const video = document.getElementById(select.dataset.previewTarget);
    video.pause(); video.src = select.value;
    const card = select.closest('[data-render-id]');
    const opt = select.selectedOptions[0];
    if (card && opt?.dataset.renderId) { // rate the attempt you are looking at
      card.dataset.renderId = opt.dataset.renderId;
      $$('input[name=render_id]', card).forEach(i => { i.value = opt.dataset.renderId; });
    }
  }));

  // -- ratings: instant, no page reload -----------------------------------------------------
  const LABELS = { '2': 'Super like', '1': 'Like', '-1': 'Dislike', '-2': 'Super dislike' };
  $$('form[data-rate]').forEach(form => {
    const status = $('.rate-status', form);
    const card = form.closest('[data-verdict]');
    let current = form.dataset.value || '0', timer;
    const paint = () => {
      $$('.rate-btn', form).forEach(b => b.setAttribute('aria-pressed', String(b.value === current)));
      if (card) card.dataset.verdict = current;
      $$('.needs-verdict', form).forEach(el => { el.hidden = current === '0'; });
      document.dispatchEvent(new CustomEvent('rokkur:rated', { detail: { form, value: current } }));
    };
    const send = async value => {
      const body = new FormData(form); body.set('value', value);
      status.textContent = 'Saving…';
      try {
        const r = await fetch(form.action, { method: 'POST', body, headers: { Accept: 'application/json' } });
        const data = await r.json();
        if (!r.ok) throw new Error(data.error || 'Could not save');
        current = String(data.value); paint();
        status.textContent = data.value ? `${data.label} saved` : 'Rating removed';
        form.closest('.queue-card')?.classList.toggle('done', data.value !== 0);
      } catch (err) { status.textContent = err.message || 'Could not save; try again'; }
    };
    form.addEventListener('submit', e => {
      e.preventDefault();
      const v = e.submitter?.value;
      if (v === undefined) return;
      send(v === current ? '0' : v); // pressing your current rating again removes it
    });
    form.addEventListener('change', e => { if (e.target.name === 'tags' && current !== '0') send(current); });
    form.addEventListener('input', e => { if (e.target.name === 'note' && current !== '0') {
      clearTimeout(timer); status.textContent = 'Unsaved note…'; timer = setTimeout(() => send(current), 900); } });
    paint();
  });

  // -- redo bar: disliked shots are suggested for a redo -------------------------------------
  const redo = $('#redo-form');
  if (redo) {
    const boxes = $$('input[name=shots]');
    const sync = () => {
      const picked = boxes.filter(b => b.checked).map(b => b.value.replace('shot_', ''));
      $('#redo-count').textContent = picked.length ? `Redo shot${picked.length > 1 ? 's' : ''} ${picked.join(', ')}` : 'Pick shots to redo';
      $('#redo-submit').disabled = !picked.length;
      redo.hidden = !picked.length && !redo.dataset.always;
    };
    document.addEventListener('rokkur:rated', e => {
      const box = $('input[name=shots]', e.detail.form.closest('.shot-card') || document.createElement('div'));
      if (box && !box.dataset.touched) box.checked = Number(e.detail.value) < 0;
      sync();
    });
    boxes.forEach(b => b.addEventListener('change', () => { b.dataset.touched = '1'; sync(); }));
    sync();
  }

  // -- shot comparison: render, original or both, kept in step -------------------------------
  $$('.shot-media').forEach(media => {
    const render = $('video[data-role=render]', media), source = $('video[data-role=source]', media);
    if (!render || !source) return;
    const start = Number(source.dataset.start || 0);
    const follow = () => { if (Math.abs(source.currentTime - (start + render.currentTime)) > 0.15) source.currentTime = start + render.currentTime; };
    render.addEventListener('play', () => { follow(); if (media.dataset.view !== 'render') source.play().catch(() => {}); });
    render.addEventListener('pause', () => source.pause());
    render.addEventListener('seeked', follow);
    render.addEventListener('timeupdate', () => { if (media.dataset.view === 'split') follow(); });
    source.addEventListener('loadedmetadata', () => { source.currentTime = start; });
    $$('.seg button', media).forEach(btn => btn.addEventListener('click', () => {
      media.dataset.view = btn.dataset.view;
      $$('.seg button', media).forEach(b => b.setAttribute('aria-pressed', String(b === btn)));
      const showRender = btn.dataset.view !== 'source', showSource = btn.dataset.view !== 'render';
      render.closest('figure').hidden = !showRender; source.closest('figure').hidden = !showSource;
      if (source.preload === 'none') source.preload = 'metadata';
      follow();
      if (!showRender) { render.pause(); }
    }));
  });

  // -- copy buttons --------------------------------------------------------------------------
  $$('[data-copy]').forEach(btn => btn.addEventListener('click', () => {
    navigator.clipboard.writeText(document.getElementById(btn.dataset.copy).textContent)
      .then(() => { btn.textContent = 'Copied'; });
  }));

  // -- live status in the rail -----------------------------------------------------------------
  const live = $('#live');
  const refreshLive = async () => {
    if (!live || document.hidden) return;
    try {
      const r = await fetch('/ui/status', { headers: { Accept: 'application/json' } });
      if (!r.ok) return;
      const d = await r.json();
      const run = d.running[0];
      $('.pulse', live).classList.toggle('on', Boolean(run));
      $('[data-live=title]', live).textContent = run ? run.name : 'Studio is idle';
      $('[data-live=detail]', live).textContent = run ? run.step : (d.queued ? `${d.queued} job${d.queued > 1 ? 's' : ''} waiting` : 'Nothing rendering');
      live.querySelector('a').href = run ? `/ui/projects/${run.project_id}` : '/ui/queue';
      const chip = $('[data-live=cloud]', live), pod = d.cloud_gpu;
      if (chip) { chip.hidden = !(pod && pod.on); chip.textContent = pod && pod.on ? pod.label : ''; }
      const badge = $('[data-count=approvals]');
      if (badge) { badge.textContent = d.approvals; badge.hidden = !d.approvals; }
    } catch { /* offline: leave the last state */ }
  };
  refreshLive(); setInterval(refreshLive, 8000);

  // -- auto refresh for pages that are working, never while you are mid-action ---------------
  const refresh = $('meta[name="studio-refresh"]');
  if (refresh) setInterval(() => {
    const playing = $$('video').some(v => !v.paused);
    const busy = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || '');
    if (!dirty && !playing && !busy && !document.hidden && !$('details[open]:not([data-keep-refresh])') && !$('dialog[open]')) {
      store.set('rokkur-scroll:' + location.pathname, String(scrollY), sessionStorage);
      location.reload();
    }
  }, Number(refresh.content) * 1000);
  const y = store.get('rokkur-scroll:' + location.pathname, sessionStorage);
  if (y) { scrollTo(0, Number(y)); store.del('rokkur-scroll:' + location.pathname, sessionStorage); }
})();
