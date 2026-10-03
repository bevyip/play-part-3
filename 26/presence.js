const EMPTY = { mimosa: [], junction: [], ducks: [], banana: [] };

export function createPresence() {
  let id = sessionStorage.getItem('gallery-visitor');
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem('gallery-visitor', id);
  }

  let timer = 0;
  let exhibit = null;
  let mobile = false;

  function post(payload) {
    fetch('/api/presence', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, ...payload }),
      keepalive: true
    }).catch(() => {});
  }

  function setExhibit(next, isMobile) {
    window.clearInterval(timer);
    timer = 0;
    if (!next) {
      if (exhibit) post({ leave: true });
      exhibit = null;
      return;
    }
    exhibit = next;
    mobile = !!isMobile;
    const beat = () => post({ exhibit, mobile });
    beat();
    timer = window.setInterval(beat, 20000);
  }

  function snapshot() {
    return fetch(`/api/presence?self=${encodeURIComponent(id)}`, { cache: 'no-store' })
      .then((res) => (res.ok ? res.json() : EMPTY))
      .catch(() => EMPTY);
  }

  window.addEventListener('pagehide', () => {
    if (exhibit) post({ leave: true });
  });

  return { setExhibit, snapshot };
}
