/* api.js — talks to the Python engine (server.py) on the same origin.
 *
 * Every call fails soft: if the server isn't running (e.g. index.html opened
 * straight from disk), health()/results() return null and trim() throws, and
 * the UI falls back to the in-browser trimmer. So the page always works.
 *
 * Exposes: window.Api.{ health, trim, results }
 */
(function (global) {
  'use strict';

  var BASE = ''; // same origin as the page

  async function health() {
    try {
      var r = await fetch(BASE + '/api/health');
      if (!r.ok) return null;
      return await r.json();
    } catch (e) { return null; }
  }

  async function trim(payload) {
    var r = await fetch(BASE + '/api/trim', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!r.ok) throw new Error('engine trim failed');
    return await r.json();
  }

  async function results() {
    try {
      var r = await fetch(BASE + '/api/results');
      if (!r.ok) return null;
      return await r.json();
    } catch (e) { return null; }
  }

  global.Api = { health: health, trim: trim, results: results };
})(window);