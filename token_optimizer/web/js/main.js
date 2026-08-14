/* main.js — entry point.
 *
 * Seeds the demo, detects the engine (server.py), wires events, and draws the
 * chart. If the engine is reachable it: enables the "live engine" controls,
 * shows an "engine connected" chip, and auto-loads results/ over the API.
 * If not, everything still works on the in-browser trimmer and manual upload.
 *
 * Load order (see index.html): trim, format, sample-data, parse, chart, api,
 * demo, then this file.
 */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };

  function seedDemo() {
    $('q').value = SampleData.question;
    $('ctx').value = SampleData.context;
    Demo.buildTicks();
    Demo.render();
  }

  function wireInputs() {
    ['q', 'ctx', 'ratio', 'price', 'rpd'].forEach(function (id) {
      $(id).addEventListener('input', Demo.render);
    });
  }

  function wireRunButton() {
    var btn = $('runbtn');
    if (!btn) return;
    btn.addEventListener('click', Demo.render);
  }

  function wireFileLoader() {
    $('loadbtn').addEventListener('click', function () { $('file').click(); });
    $('file').addEventListener('change', function (e) {
      var file = e.target.files[0];
      if (!file) return;
      var reader = new FileReader();
      reader.onload = function () {
        var note = $('loadnote');
        try {
          Chart.load(ParseResults.parse(JSON.parse(reader.result)));
          note.textContent = 'loaded ' + file.name + ' \u2014 showing your real numbers';
          note.style.color = 'var(--mint)';
        } catch (err) {
          note.textContent = "couldn't read that file \u2014 expected curve.json or bench.json";
          note.style.color = 'var(--red)';
        }
      };
      reader.readAsText(file);
    });
  }

  function setChip(state, text) {
    var chip = $('enginechip');
    if (!chip) return;
    chip.textContent = text;
    chip.className = 'chip ' + state; // chip on | chip off
  }

  function wireEngineControls(health) {
    var row = $('enginerow');
    var toggle = $('engtoggle');
    var relsel = $('relsel');
    if (row) row.hidden = false;

    // Enable embedding option only if the server reports it available.
    if (relsel) {
      var embOpt = relsel.querySelector('option[value="embedding"]');
      if (embOpt && !health.embeddings) {
        embOpt.disabled = true;
        embOpt.textContent = 'embedding (install sentence-transformers)';
      }
    }

    toggle.checked = true;
    Demo.setEngine({ available: true, on: true, relevance: 'lexical' });

    toggle.addEventListener('change', function () {
      Demo.setEngine({ on: toggle.checked });
    });
    if (relsel) {
      relsel.addEventListener('change', function () {
        Demo.setEngine({ relevance: relsel.value });
      });
    }
  }

  async function connectEngine() {
    var health = await Api.health();
    if (!health) {
      setChip('off', 'engine: offline \u00b7 using browser estimate');
      return;
    }
    setChip('on', 'engine: connected');
    wireEngineControls(health);

    // Auto-load real results if the harness has written any.
    var res = await Api.results();
    if (res && (res.bench || res.curve)) {
      var data = ParseResults.parse(res.bench || res.curve);
      Chart.load(data);
      var note = $('loadnote');
      if (note) {
        note.textContent = 'auto-loaded from results/ \u2014 live from the engine';
        note.style.color = 'var(--mint)';
      }
    }
  }

  function init() {
    seedDemo();
    wireInputs();
    wireRunButton();
    wireFileLoader();
    Chart.load(SampleData.benchmark); // sample first; engine may replace it
    connectEngine();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();