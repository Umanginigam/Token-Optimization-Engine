/* trim.js — the relevance trimmer, ported 1:1 from harness/trim.py.
 *
 * This is the real algorithm, not a mock: TF-IDF cosine similarity between the
 * question and each context sentence, keeping the top `keepRatio` fraction and
 * reassembling in original reading order. Verified against the Python version.
 *
 * Exposes: window.Trim.{ trimContext, splitSentences, scoreSentences }
 */
(function (global) {
  'use strict';

  function normalize(s) {
    return s.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .replace(/\b(a|an|the)\b/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function tokenize(s) {
    return normalize(s).split(' ').filter(Boolean);
  }

  function splitSentences(text) {
    return text.split(/(?<=[.!?])\s+/).map(function (x) { return x.trim(); }).filter(Boolean);
  }

  /* IDF over the sentences of THIS context, so rare words weigh more. */
  function makeIdf(docs) {
    var N = docs.length, df = {};
    docs.forEach(function (d) {
      new Set(d).forEach(function (w) { df[w] = (df[w] || 0) + 1; });
    });
    return function (w) { return Math.log((N + 1) / ((df[w] || 0) + 1)) + 1; };
  }

  function scoreSentences(question, sentences) {
    var docs = sentences.map(tokenize);
    var q = tokenize(question);
    var idf = makeIdf(docs);

    var qtf = {};
    q.forEach(function (w) { qtf[w] = (qtf[w] || 0) + 1; });
    var qvec = {};
    Object.keys(qtf).forEach(function (w) { qvec[w] = qtf[w] * idf(w); });
    var qnorm = Math.sqrt(Object.values(qvec).reduce(function (a, v) { return a + v * v; }, 0)) || 1;

    return docs.map(function (d) {
      var tf = {};
      d.forEach(function (w) { tf[w] = (tf[w] || 0) + 1; });
      var dv = {};
      Object.keys(tf).forEach(function (w) { dv[w] = tf[w] * idf(w); });

      var dot = 0;
      Object.keys(qvec).forEach(function (w) { if (dv[w]) dot += qvec[w] * dv[w]; });

      var dnorm = Math.sqrt(Object.values(dv).reduce(function (a, v) { return a + v * v; }, 0)) || 1;
      return dot / (qnorm * dnorm);
    });
  }

  /* Returns [{ s, keep }] in original order, so the UI can show kept vs dropped. */
  function trimContext(question, context, keepRatio) {
    var S = splitSentences(context);
    if (S.length <= 1) {
      return S.map(function (s) { return { s: s, keep: true }; });
    }
    var scores = scoreSentences(question, S);
    var k = Math.min(S.length, Math.max(1, Math.ceil(S.length * keepRatio)));
    var top = Array.from(scores.keys())
      .sort(function (a, b) { return scores[b] - scores[a]; })
      .slice(0, k);
    var keepSet = new Set(top);
    return S.map(function (s, i) { return { s: s, keep: keepSet.has(i) }; });
  }

  global.Trim = {
    trimContext: trimContext,
    splitSentences: splitSentences,
    scoreSentences: scoreSentences
  };
})(window);