// VISIONAI Reader service worker — makes repeat visits feel instant.
// Shell + images are cached aggressively; API JSON uses stale-while-revalidate.

const VERSION = 'vr-v1';
const SHELL = 'vr-shell-' + VERSION;
const IMGS = 'vr-img-' + VERSION;
const API_CACHE = 'vr-api-' + VERSION;

const SHELL_FILES = ['./reader.html', './manifest.webmanifest'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(SHELL_FILES)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => ![SHELL, IMGS, API_CACHE].includes(k)).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const u = new URL(req.url);

  // 1. manga page images + covers: cache-first (they never change)
  if (/mangadex\.network|uploads\.mangadex\.org/.test(u.hostname)) {
    e.respondWith(
      caches.open(IMGS).then(async (c) => {
        const hit = await c.match(req);
        if (hit) return hit;
        const res = await fetch(req);
        if (res.ok) c.put(req, res.clone());
        return res;
      }).catch(() => fetch(req))
    );
    return;
  }

  // 2. the API proxy: serve cached copy instantly, refresh in background
  if (u.pathname.includes('/functions/mdProxy')) {
    e.respondWith(
      caches.open(API_CACHE).then(async (c) => {
        const hit = await c.match(req);
        const net = fetch(req).then((res) => {
          if (res.ok) c.put(req, res.clone());
          return res;
        }).catch(() => null);
        if (hit) { e.waitUntil(net); return hit; }
        const res = await net;
        return res || new Response(JSON.stringify({ error: 'offline' }), { headers: { 'Content-Type': 'application/json' } });
      })
    );
    return;
  }

  // 3. the app shell: cache-first, fall back to network
  if (u.origin === self.location.origin && u.pathname.endsWith('reader.html')) {
    e.respondWith(
      caches.open(SHELL).then(async (c) => (await c.match('./reader.html')) || fetch(req).then((res) => {
        if (res.ok) c.put('./reader.html', res.clone());
        return res;
      }))
    );
  }
});
