/* format.js — small display helpers shared across the UI.
 *
 * Exposes: window.Fmt.{ estimateTokens, number, money, escapeHtml }
 */
(function (global) {
  'use strict';

  /* Rough local token estimate (~1.3 tokens per word). The harness itself
     prefers the token counts the API reports; this is only for the live demo. */
  function estimateTokens(s) {
    if (!s || !s.trim()) return 0;
    return Math.max(1, Math.round(s.trim().split(/\s+/).length * 1.3));
  }

  function number(n) {
    return n.toLocaleString('en-US');
  }

  function money(n) {
    if (n >= 1000) return '$' + (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k';
    return '$' + n.toFixed(n < 10 ? 2 : 0);
  }

  function escapeHtml(t) {
    return t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  global.Fmt = {
    estimateTokens: estimateTokens,
    number: number,
    money: money,
    escapeHtml: escapeHtml
  };
})(window);