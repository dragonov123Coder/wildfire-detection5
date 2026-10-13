(function () {
  const grid = document.getElementById('unit-grid');
  const emptyHint = document.getElementById('empty-hint');
  const overlay = document.getElementById('overlay');
  const addBtn = document.getElementById('add-unit-btn');
  const cancelBtn = document.getElementById('cancel-btn');
  const submitBtn = document.getElementById('submit-btn');
  const doneBtn = document.getElementById('done-btn');
  const copyBtn = document.getElementById('copy-btn');
  const formStep = document.getElementById('form-step');
  const resultStep = document.getElementById('result-step');
  const formError = document.getElementById('form-error');
  const configOutput = document.getElementById('config-output');
  const fName = document.getElementById('f-name');
  const fLat = document.getElementById('f-lat');
  const fLon = document.getElementById('f-lon');

  function timeAgo(iso) {
    if (!iso) return 'never';
    const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
    if (seconds < 5) return 'just now';
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
  }

  function statusLabel(unit) {
    if (unit.connected) return 'Online';
    if (unit.status === 'awaiting_connection') return 'Awaiting connection';
    return 'Offline';
  }

  function statusClass(unit) {
    if (unit.connected) return 'online';
    if (unit.status === 'awaiting_connection') return 'awaiting_connection';
    return 'offline';
  }

  function locationText(unit) {
    if (unit.lat === null || unit.lat === undefined || unit.lon === null || unit.lon === undefined) {
      return 'Location unknown';
    }
    return `${Number(unit.lat).toFixed(3)}, ${Number(unit.lon).toFixed(3)}`;
  }

  // Cards are created once (playing the entrance animation once) and then
  // updated in place on every poll, so the list doesn't flicker or re-animate.
  const cards = new Map(); // unit_id -> { el, nameEl, statusEl, metaEl, deleteBtn }

  function createCard(unit) {
    const el = document.createElement('div');
    el.className = 'unit-card card fade-rise';
    el.innerHTML = `
      <div class="unit-head">
        <div>
          <div class="unit-name"></div>
          <div class="unit-id">${escapeHtml(unit.unit_id)}</div>
        </div>
      </div>
      <div class="status-row"></div>
      <div class="meta-row"></div>
      <div class="card-actions">
        <a class="btn" href="/units/${encodeURIComponent(unit.unit_id)}">Dashboard</a>
        <button class="btn btn-danger delete-btn" type="button" data-unit-id="${escapeHtml(unit.unit_id)}" title="Delete unit">${window.TRASH_ICON || ''} Delete</button>
      </div>
    `;
    return {
      el,
      nameEl: el.querySelector('.unit-name'),
      statusEl: el.querySelector('.status-row'),
      metaEl: el.querySelector('.meta-row'),
      deleteBtn: el.querySelector('.delete-btn'),
      lastStatus: null,
    };
  }

  function updateCard(card, unit) {
    card.nameEl.textContent = unit.name;
    card.deleteBtn.dataset.unitName = unit.name;
    const statusHtml = `<span class="status-dot ${statusClass(unit)}"></span>${statusLabel(unit)} &middot; last seen ${timeAgo(unit.last_seen)}`;
    if (statusHtml !== card.lastStatus) {
      card.statusEl.innerHTML = statusHtml;
      card.lastStatus = statusHtml;
    }
    card.metaEl.textContent = locationText(unit);
  }

  function renderUnits(units) {
    emptyHint.hidden = units.length > 0;

    const seen = new Set(units.map((u) => u.unit_id));
    for (const [unitId, card] of cards) {
      if (!seen.has(unitId)) {
        card.el.remove();
        cards.delete(unitId);
      }
    }

    units.forEach((unit, i) => {
      let card = cards.get(unit.unit_id);
      if (!card) {
        card = createCard(unit);
        cards.set(unit.unit_id, card);
      }
      updateCard(card, unit);
      // Only touch the DOM when a card is not already at its slot; moving an
      // element restarts its CSS animation.
      if (grid.children[i] !== card.el) grid.insertBefore(card.el, grid.children[i] || null);
    });
  }

  function swalTheme() {
    const css = getComputedStyle(document.documentElement);
    const v = (name) => css.getPropertyValue(name).trim();
    return {
      background: v('--surface-1'),
      color: v('--text-primary'),
      confirmButtonColor: v('--danger'),
      cancelButtonColor: v('--text-muted'),
    };
  }

  async function confirmAndDelete(unitId, unitName) {
    const result = await Swal.fire({
      ...swalTheme(),
      title: 'Delete this unit?',
      html: `<b>${escapeHtml(unitName)}</b> (<code>${escapeHtml(unitId)}</code>) will be removed from the Unit Manager.<br><br>` +
            `If its Raspberry Pi is still configured with this id and connects again, it will re-register as a new unnamed unit.`,
      icon: 'warning',
      showCancelButton: true,
      confirmButtonText: 'Delete',
      cancelButtonText: 'Cancel',
      focusCancel: true,
    });
    if (!result.isConfirmed) return;

    try {
      const resp = await fetch(`/api/units/${encodeURIComponent(unitId)}`, { method: 'DELETE' });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        await Swal.fire({ ...swalTheme(), title: 'Could not delete unit', text: body.error || resp.statusText, icon: 'error', confirmButtonColor: swalTheme().cancelButtonColor });
        return;
      }
      await Swal.fire({ ...swalTheme(), title: 'Unit deleted', icon: 'success', timer: 1200, showConfirmButton: false });
      refresh();
    } catch (e) {
      await Swal.fire({ ...swalTheme(), title: 'Request failed', text: e.message, icon: 'error', confirmButtonColor: swalTheme().cancelButtonColor });
    }
  }

  // Cards are re-rendered every few seconds, so delegate from the grid.
  grid.addEventListener('click', (e) => {
    const btn = e.target.closest('.delete-btn');
    if (btn) confirmAndDelete(btn.dataset.unitId, btn.dataset.unitName);
  });

  function escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s == null ? '' : String(s);
    return div.innerHTML.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  async function refresh() {
    try {
      const resp = await fetch('/api/units');
      if (!resp.ok) return;
      const units = await resp.json();
      renderUnits(units);
    } catch (e) {
      // Keep last-known list on transient network errors.
    }
  }

  function openModal(prefill) {
    formStep.hidden = false;
    resultStep.hidden = true;
    formError.textContent = '';
    fName.value = '';
    fLat.value = prefill && prefill.lat !== undefined ? prefill.lat : '';
    fLon.value = prefill && prefill.lon !== undefined ? prefill.lon : '';
    overlay.hidden = false;
  }

  function closeModal() {
    overlay.hidden = true;
  }

  addBtn.addEventListener('click', () => openModal());
  cancelBtn.addEventListener('click', closeModal);
  doneBtn.addEventListener('click', () => { closeModal(); refresh(); });
  overlay.addEventListener('click', (e) => { if (e.target === overlay) closeModal(); });

  submitBtn.addEventListener('click', async () => {
    const lat = parseFloat(fLat.value);
    const lon = parseFloat(fLon.value);
    if (Number.isNaN(lat) || Number.isNaN(lon)) {
      formError.textContent = 'Latitude and longitude are required.';
      return;
    }
    submitBtn.disabled = true;
    try {
      const resp = await fetch('/api/units', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: fName.value.trim(), lat, lon }),
      });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        formError.textContent = body.error || 'Failed to create unit.';
        return;
      }
      formStep.hidden = true;
      resultStep.hidden = false;
      configOutput.textContent = JSON.stringify(body.config_snippet, null, 2);
    } catch (e) {
      formError.textContent = 'Request failed: ' + e.message;
    } finally {
      submitBtn.disabled = false;
    }
  });

  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(configOutput.textContent);
      copyBtn.textContent = 'Copied!';
      setTimeout(() => { copyBtn.innerHTML = copyBtn.dataset.original; }, 1500);
    } catch (e) {
      // Clipboard API unavailable (e.g. non-HTTPS context) -- selection fallback.
      const range = document.createRange();
      range.selectNodeContents(configOutput);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
  });
  copyBtn.dataset.original = copyBtn.innerHTML;

  // If arriving from the prediction map with a chosen point, open the
  // add-unit form pre-filled with that location.
  const params = new URLSearchParams(window.location.search);
  if (params.has('lat') && params.has('lon')) {
    openModal({ lat: params.get('lat'), lon: params.get('lon') });
  }

  refresh();
  setInterval(refresh, 4000);
})();
