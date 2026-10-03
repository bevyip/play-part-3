/**
 * Snapshot presence for the overhead gallery, stored in a Supabase table.
 * Set SUPABASE_URL and SUPABASE_SECRET_KEY on the Vercel project.
 * Without those, the room still loads and the crowd is empty.
 */
const TTL_MS = 45000;
const ID_RE = /^[0-9a-f-]{36}$/i;
const EXHIBITS = new Set(['mimosa', 'junction', 'ducks', 'banana']);

function emptyCrowd() {
  return { mimosa: [], junction: [], ducks: [], banana: [] };
}

function configured() {
  return Boolean(process.env.SUPABASE_URL && process.env.SUPABASE_SECRET_KEY);
}

async function table(method, query, body, prefer) {
  const key = process.env.SUPABASE_SECRET_KEY;
  const headers = { apikey: key, 'Content-Type': 'application/json' };
  // Legacy service_role keys are JWTs and must also be sent as a bearer token.
  if (key.startsWith('eyJ')) headers.Authorization = `Bearer ${key}`;
  if (prefer) headers.Prefer = prefer;
  const base = process.env.SUPABASE_URL.replace(/\/+$/, '');
  const res = await fetch(`${base}/rest/v1/presence${query}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined
  });
  if (!res.ok) throw new Error(`supabase ${res.status}: ${await res.text()}`);
  return method === 'GET' ? res.json() : null;
}

function readBody(req) {
  if (!req.body) return {};
  if (typeof req.body === 'string') {
    try {
      return JSON.parse(req.body);
    } catch (err) {
      return {};
    }
  }
  return req.body;
}

module.exports = async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');

  if (req.method === 'GET') {
    const crowd = emptyCrowd();
    if (!configured()) return res.status(200).json(crowd);
    try {
      const self = String(req.query.self || '');
      const cutoff = encodeURIComponent(new Date(Date.now() - TTL_MS).toISOString());
      const [rows] = await Promise.all([
        table('GET', `?select=id,exhibit,mobile&seen_at=gte.${cutoff}`),
        table('DELETE', `?seen_at=lt.${cutoff}`)
      ]);
      for (const row of rows) {
        if (row.id !== self && crowd[row.exhibit]) crowd[row.exhibit].push(!!row.mobile);
      }
      return res.status(200).json(crowd);
    } catch (err) {
      console.error(err);
      return res.status(200).json(crowd);
    }
  }

  if (req.method === 'POST') {
    const body = readBody(req);
    const id = String(body.id || '');
    if (!ID_RE.test(id)) return res.status(400).json({ ok: false });
    if (!configured()) return res.status(200).json({ ok: true, stored: false, reason: 'not-configured' });
    try {
      if (body.leave) {
        await table('DELETE', `?id=eq.${id}`);
      } else if (EXHIBITS.has(body.exhibit)) {
        await table(
          'POST',
          '',
          { id, exhibit: body.exhibit, mobile: !!body.mobile, seen_at: new Date().toISOString() },
          'resolution=merge-duplicates,return=minimal'
        );
      } else {
        return res.status(400).json({ ok: false });
      }
      return res.status(200).json({ ok: true, stored: true });
    } catch (err) {
      console.error(err);
      return res.status(200).json({ ok: true, stored: false, reason: 'database-error' });
    }
  }

  res.setHeader('Allow', 'GET, POST');
  return res.status(405).end();
};
