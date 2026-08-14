/* demo.js — the live trimmer.
 *
 * Two paths, same paint routine:
 *   - LOCAL: the in-browser trimmer (trim.js) with an estimated token count.
 *            Instant, always available.
 *   - ENGINE: when server.py is running and "Use live engine" is on, a debounced
 *            call to /api/trim returns the REAL tokenizer count and the chosen
 *            relevance (lexical or embedding). It overwrites the local estimate.
 *
 * Local render fires first for snappy feedback; the engine result reconciles a
 * beat later. If the engine call fails, the local numbers simply stay.
 *
 * Depends on: trim.js, format.js, api.js
 * Exposes: window.Demo.{ render, buildTicks, setEngine, engineState }
 */
(function (global) {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };

  var engine = { on: false, available: false, relevance: 'lexical' };
  var syncTimer = null;

  function buildTicks() {
    var t = $('ticks');
    if (!t || t.childElementCount) return;
    for (var i = 0; i < 40; i++) t.appendChild(document.createElement('i'));
  }

  function renderProjector(before, after) {
    var price = parseFloat($('price').value) || 0;
    var rpd = parseFloat(($('rpd').value || '').replace(/[, ]/g, '')) || 0;
    var monthlyRequests = rpd * 30;

    var costBefore = monthlyRequests * before / 1e6 * price;
    var costAfter = monthlyRequests * after / 1e6 * price;
    var saved = Math.max(0, costBefore - costAfter);

    $('spendbefore').textContent = Fmt.money(costBefore);
    $('spendafter').textContent = Fmt.money(costAfter);
    $('spendsaved').innerHTML = Fmt.money(saved) + '<small>/mo</small>';
  }

  /* One paint routine for both paths. `sentences` = [{text, keep}]. */
  function paint(sentences, before, after, source) {
    var savedTokens = Math.max(0, before - after);
    var savedPct = before ? savedTokens / before : 0;

    $('meterkept').style.width = (before ? (after / before * 100) : 100) + '%';
    $('metercap').textContent = savedPct > 0
      ? ('\u2212' + Math.round(savedPct * 100) + '% sent')
      : 'full context';
    $('savedpct').textContent = (savedPct > 0 ? '\u2212' : '') + Math.round(savedPct * 100) + '% tokens';

    $('tbefore').innerHTML = Fmt.number(before) + ' <small>tok</small>';
    $('tafter').innerHTML = Fmt.number(after) + ' <small>tok</small>';
    $('tsaved').innerHTML = Fmt.number(savedTokens) + ' <small>tok</small>';

    $('outbox').innerHTML = sentences.map(function (p) {
      return '<span class="' + (p.keep ? 's-keep' : 's-drop') + '">' +
             Fmt.escapeHtml(p.text) + '</span>';
    }).join(' ');

    var src = $('demosource');
    if (src) src.textContent = source;

    renderProjector(before, after);
  }

  function localRender() {
    var question = $('q').value;
    var context = $('ctx').value;
    var ratio = (+$('ratio').value) / 100;

    $('ratioval').textContent = Math.round(ratio * 100) + '%';
    $('ratio').style.setProperty('--fill', (ratio * 100) + '%');

    var parts = Trim.trimContext(question, context, ratio)
      .map(function (p) { return { text: p.s, keep: p.keep }; });
    var before = Fmt.estimateTokens(context);
    var kept = parts.filter(function (p) { return p.keep; })
                    .map(function (p) { return p.text; }).join(' ');
    var after = Fmt.estimateTokens(kept);

    var label = (engine.on && engine.available) ? 'estimate \u00b7 syncing engine\u2026' : 'local estimate';
    paint(parts, before, after, label);
  }

  async function syncEngine() {
    if (!(engine.on && engine.available)) return;
    var payload = {
      question: $('q').value,
      context: $('ctx').value,
      keep_ratio: (+$('ratio').value) / 100,
      relevance: engine.relevance
    };
    try {
      var res = await Api.trim(payload);
      var sentences = res.sentences.map(function (s) { return { text: s.text, keep: s.keep }; });
      paint(sentences, res.tokens_before, res.tokens_after,
            'live engine \u00b7 ' + res.relevance_used + ' \u00b7 real tokens');
    } catch (e) {
      /* leave the local estimate in place */
    }
  }

  function flashRan() {
    var card = $('outcard');
    if (!card) return;
    card.classList.remove('just-ran');
    // eslint-disable-next-line no-unused-expressions
    void card.offsetWidth; // restart the animation even if it's already running
    card.classList.add('just-ran');
  }

  function render() {
    localRender();
    flashRan();
    if (engine.on && engine.available) {
      clearTimeout(syncTimer);
      syncTimer = setTimeout(syncEngine, 250);
    }
  }

  function setEngine(patch) {
    Object.assign(engine, patch);
    render();
  }

  global.Demo = {
    render: render,
    buildTicks: buildTicks,
    setEngine: setEngine,
    engineState: function () { return Object.assign({}, engine); }
  };
})(window);