/* sample-data.js — what the page shows before you load your own results.
 *
 * SAMPLE holds real numbers from a HotpotQA semantic-cache threshold sweep
 * (baseline F1 67.08). Replace it by loading your own curve.json / bench.json
 * through the UI — nothing else needs to change.
 *
 * Exposes: window.SampleData.{ question, context, benchmark }
 */
(function (global) {
  'use strict';

  var QUESTION = 'In what year did the Eiffel Tower officially open to the public?';

  var CONTEXT =
    'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. ' +
    'It is named after the engineer Gustave Eiffel, whose company designed and built the tower. ' +
    "Locals initially criticized it as an eyesore that clashed with the city's classical architecture. " +
    'Construction began in 1887 and took just over two years to complete. ' +
    "The tower officially opened to the public in 1889 as the entrance arch to the World's Fair. " +
    'It stands 330 metres tall, roughly the height of an 81-storey building. ' +
    'It was the tallest man-made structure in the world for 41 years until the Chrysler Building surpassed it. ' +
    'Today it is one of the most visited paid monuments on the planet, drawing millions of visitors each year.';

  var BENCHMARK = {
    kind: 'cache',
    title: 'HotpotQA semantic-cache threshold sweep',
    baselineF1: 67.08,
    points: [
      { label: '0.99', saved: 68.5, f1: 67.19, d: +0.11, sig: 'ns' },
      { label: '0.95', saved: 76.6, f1: 66.86, d: -0.23, sig: 'ns', rec: true },
      { label: '0.90', saved: 77.6, f1: 66.64, d: -0.44, sig: 'ns' },
      { label: '0.85', saved: 77.6, f1: 66.64, d: -0.44, sig: 'ns' },
      { label: '0.80', saved: 77.6, f1: 66.64, d: -0.44, sig: 'ns' }
    ],
    table: [
      { cfg: 'baseline',   saved: 0.0,  d: 0.00,  sig: 'base' },
      { cfg: 'cache 0.99', saved: 68.5, d: +0.11, sig: 'ns' },
      { cfg: 'cache 0.95', saved: 76.6, d: -0.23, sig: 'ns' },
      { cfg: 'cache 0.90', saved: 77.6, d: -0.44, sig: 'ns' }
    ]
  };

  global.SampleData = {
    question: QUESTION,
    context: CONTEXT,
    benchmark: BENCHMARK
  };
})(window);