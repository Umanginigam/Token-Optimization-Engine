/* chart.js — renders the savings-vs-quality scatter and the benchmark table.
 *
 * The chart is hand-built SVG (no chart library) so it inherits the CSS
 * variables from tokens.css and stays theme-consistent.
 *
 * Exposes: window.Chart.{ drawChart, drawTable, load }
 */
(function (global) {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };

  function drawChart(data) {
    var W = 560, H = 340, pad = { l: 52, r: 24, t: 22, b: 46 };

    var xs = data.points.map(function (p) { return p.saved; });
    var ys = data.points.map(function (p) { return p.f1; });
    var allY = ys.concat([data.baselineF1]);

    var ymin = Math.min.apply(null, allY);
    var ymax = Math.max.apply(null, allY);
    var yr = (ymax - ymin) || 1;
    ymin -= yr * 0.35;
    ymax += yr * 0.35;

    var xmin = 0;
    var xmax = Math.max(100, Math.ceil(Math.max.apply(null, xs) / 10) * 10);

    var X = function (v) { return pad.l + (v - xmin) / (xmax - xmin) * (W - pad.l - pad.r); };
    var Y = function (v) { return H - pad.b - (v - ymin) / (ymax - ymin) * (H - pad.t - pad.b); };

    var g = '';

    /* horizontal grid + y labels */
    for (var i = 0; i <= 4; i++) {
      var yy = pad.t + i * (H - pad.t - pad.b) / 4;
      var val = ymax - i * (ymax - ymin) / 4;
      g += '<line x1="' + pad.l + '" y1="' + yy + '" x2="' + (W - pad.r) + '" y2="' + yy +
           '" stroke="var(--line)" stroke-width="1"/>';
      g += '<text x="' + (pad.l - 9) + '" y="' + (yy + 4) + '" text-anchor="end" fill="var(--faint)" ' +
           'font-family="var(--mono)" font-size="10">' + val.toFixed(0) + '</text>';
    }

    /* x labels */
    var step = Math.round(xmax / 5 / 5) * 5 || 20;
    for (var v = 0; v <= xmax; v += step) {
      g += '<text x="' + X(v) + '" y="' + (H - pad.b + 22) + '" text-anchor="middle" fill="var(--faint)" ' +
           'font-family="var(--mono)" font-size="10">' + v + '%</text>';
    }

    /* baseline reference */
    g += '<line x1="' + pad.l + '" y1="' + Y(data.baselineF1) + '" x2="' + (W - pad.r) +
         '" y2="' + Y(data.baselineF1) + '" stroke="var(--muted)" stroke-width="1" stroke-dasharray="4 4"/>';
    g += '<text x="' + (W - pad.r) + '" y="' + (Y(data.baselineF1) - 7) + '" text-anchor="end" ' +
         'fill="var(--muted)" font-family="var(--mono)" font-size="10">baseline F1 ' +
         data.baselineF1.toFixed(1) + '</text>';

    /* connecting line, left to right by savings */
    var sorted = data.points.slice().sort(function (a, b) { return a.saved - b.saved; });
    g += '<polyline points="' + sorted.map(function (p) { return X(p.saved) + ',' + Y(p.f1); }).join(' ') +
         '" fill="none" stroke="var(--mint-line)" stroke-width="1.5"/>';

    /* points — amber marks a statistically real quality cost */
    data.points.forEach(function (p) {
      var col = p.sig === 'real' ? 'var(--amber)' : 'var(--mint)';
      if (p.rec) {
        g += '<circle cx="' + X(p.saved) + '" cy="' + Y(p.f1) +
             '" r="11" fill="none" stroke="var(--mint)" stroke-width="1.5" opacity=".55"/>';
      }
      g += '<circle cx="' + X(p.saved) + '" cy="' + Y(p.f1) + '" r="5.5" fill="' + col + '"/>';
      g += '<text x="' + X(p.saved) + '" y="' + (Y(p.f1) - 13) + '" text-anchor="middle" fill="var(--muted)" ' +
           'font-family="var(--mono)" font-size="9.5">' + p.label + '</text>';
    });

    /* axis titles */
    g += '<text x="' + (pad.l + (W - pad.l - pad.r) / 2) + '" y="' + (H - 6) + '" text-anchor="middle" ' +
         'fill="var(--muted)" font-family="var(--mono)" font-size="10" letter-spacing="1">TOKENS SAVED →</text>';
    g += '<text transform="translate(14,' + (pad.t + (H - pad.t - pad.b) / 2) + ') rotate(-90)" ' +
         'text-anchor="middle" fill="var(--muted)" font-family="var(--mono)" font-size="10" ' +
         'letter-spacing="1">F1 QUALITY</text>';

    $('chart').innerHTML =
      '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" ' +
      'aria-label="Savings versus quality scatter">' + g + '</svg>';
  }

  function drawTable(data) {
    var rows = data.table || [];
    $('benchbody').innerHTML = rows.map(function (r) {
      var pill = r.sig === 'base' ? '<span class="pill base">baseline</span>'
               : r.sig === 'real' ? '<span class="pill real">real cost</span>'
               : '<span class="pill ns">safe · ns</span>';
      var dtxt = r.sig === 'base' ? '—' : (r.d >= 0 ? '+' : '') + r.d.toFixed(2);
      return '<tr><td class="cfg">' + r.cfg + '</td><td>' + r.saved.toFixed(1) +
             '%</td><td>' + dtxt + '</td><td>' + pill + '</td></tr>';
    }).join('');
  }

  function load(data) {
    drawChart(data);
    drawTable(data);
    $('chartfoot').textContent =
      (data.kind === 'bench' ? 'your data · ' : 'sample: ') + (data.title || 'results');
  }

  global.Chart = { drawChart: drawChart, drawTable: drawTable, load: load };
})(window);