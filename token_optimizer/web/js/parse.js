/* parse.js — read the JSON files the harness writes and normalize them
 * into the shape the chart and table expect.
 *
 * Handles both:
 *   bench.json  (Phase 5) — array of { config, tokens_saved_pct, f1,
 *                                      f1_delta, significance }
 *   curve.json  (Phase 3) — { baseline:{f1}, points:[{keep_ratio,
 *                             tokens_saved_pct, f1, f1_delta}], recommended }
 *
 * Throws if the file is neither, so the UI can show a clear error.
 *
 * Exposes: window.ParseResults.parse(obj) -> chartData
 */
(function (global) {
  'use strict';

  function fromBench(rows) {
    var base = rows.find(function (r) { return r.significance === 'base'; }) || rows[0];

    var points = rows
      .filter(function (r) { return r.significance !== 'base'; })
      .map(function (r) {
        return {
          label: r.config,
          saved: r.tokens_saved_pct,
          f1: r.f1,
          d: r.f1_delta,
          sig: r.significance === 'ns' ? 'ns' : 'real'
        };
      });

    var table = rows.map(function (r) {
      return {
        cfg: r.config,
        saved: r.tokens_saved_pct,
        d: r.f1_delta || 0,
        sig: r.significance === 'base' ? 'base' : (r.significance === 'ns' ? 'ns' : 'real')
      };
    });

    return { kind: 'bench', title: 'Phase 5 benchmark', baselineF1: base.f1, points: points, table: table };
  }

  function fromCurve(obj) {
    var baseF1 = (obj.baseline && obj.baseline.f1) || obj.points[0].f1;
    var rec = obj.recommended;

    var points = obj.points
      .filter(function (p) { return p.tokens_saved_pct > 0.01; })
      .map(function (p) {
        return {
          label: p.keep_ratio != null ? p.keep_ratio.toFixed(2) : '',
          saved: p.tokens_saved_pct,
          f1: p.f1,
          d: p.f1_delta || 0,
          sig: 'ns',
          rec: !!(rec && Math.abs(p.keep_ratio - rec.keep_ratio) < 1e-6)
        };
      });

    var table = obj.points.map(function (p) {
      return {
        cfg: p.keep_ratio >= 1 ? 'baseline' : 'keep ' + p.keep_ratio.toFixed(2),
        saved: p.tokens_saved_pct,
        d: p.f1_delta || 0,
        sig: p.keep_ratio >= 1 ? 'base' : 'ns'
      };
    });

    return { kind: 'bench', title: 'your trim/cache sweep', baselineF1: baseF1, points: points, table: table };
  }

  function parse(obj) {
    if (Array.isArray(obj) && obj[0] && 'config' in obj[0]) return fromBench(obj);
    if (obj && obj.points) return fromCurve(obj);
    throw new Error('unrecognized results file');
  }

  global.ParseResults = { parse: parse };
})(window);