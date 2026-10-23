// MiniMe showcase: fills the page from data.js (written by scripts/build_showcase.py),
// draws the frontier, and loads the chosen samples from their public object.
(() => {
  "use strict";

  const D = window.MINIME;
  const MINUS = "−";
  const SVG_NS = "http://www.w3.org/2000/svg";

  // --- formatting --------------------------------------------------------

  const signed = (x, places) => (x < 0 ? MINUS : "+") + Math.abs(x).toFixed(places);
  const FORMATS = {
    raw: (x) => String(x),
    int: (x) => x.toLocaleString("en-US"),
    word: (x) => ["No", "One", "Two", "Three", "Four", "Five"][x] ?? String(x),
    floor: (x) => String(Math.floor(x)),
    f1: (x) => x.toFixed(1),
    f2: (x) => x.toFixed(2),
    f3: (x) => x.toFixed(3),
    f4: (x) => x.toFixed(4),
    s3: (x) => signed(x, 3),
    s4: (x) => signed(x, 4),
    M0: (x) => `${Math.round(x / 1e6)}M`,
    M1: (x) => `${(x / 1e6).toFixed(1)}M`,
    sci: (x) => x.toExponential(0).replace("e-", "e−"),
    date: (x) => String(x).slice(0, 10),
    pct1: (x) => `${(100 * x).toFixed(1)}%`,
    pm2: (c) => `${signed(c.difference, 2)} ± ${c.half_width.toFixed(2)}`,
    pm3: (c) => `${signed(c.difference, 3)} ± ${c.half_width.toFixed(3)}`,
    ratio: (parts) => {
      const order = ["web", "dialogue", "code"];
      const unit = Math.min(...order.map((k) => parts[k]));
      return order.map((k) => Math.round(parts[k] / unit)).join(" : ");
    },
  };

  const get = (path) => path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), D);

  function bind() {
    for (const el of document.querySelectorAll("[data-v]")) {
      const v = get(el.dataset.v);
      el.textContent = v == null ? "—" : FORMATS[el.dataset.f || "raw"](v);
    }
  }

  const el = (tag, attrs = {}, parent = null, text = null) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text != null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  };
  const sv = (tag, attrs = {}, parent = null) => {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  };

  // --- the frontier ------------------------------------------------------

  // Group -> how it is drawn and where its direct label sits (dx, dy, anchor).
  const STYLE = {
    adaptation: { cls: "final", shape: "circle", r: 7, label: "Flagship + C", sub: "final model", place: [12, -7, "start"] },
    flagship: { cls: "flagship", shape: "square", r: 6, label: "Flagship", place: [12, 4, "start"] },
    P4: { cls: "probe", shape: "circle", r: 5, label: "P4", place: [10, 4, "start"] },
    "E1 25/75": { cls: "probe", shape: "circle", r: 5, label: "E1 25/75", place: [-10, 16, "end"] },
    "E1 50/50": { cls: "probe", shape: "circle", r: 5, label: "E1 50/50", place: [-10, 14, "end"] },
    P1: { cls: "probe", shape: "circle", r: 5, label: "P1", offscale: true },
  };
  const ORDER = ["P1", "P4", "E1 25/75", "E1 50/50", "flagship", "adaptation"];

  let split = "test";
  let lastWidth = 0;

  function niceStep(span, target) {
    const raw = span / target;
    const pow = 10 ** Math.floor(Math.log10(raw));
    return [1, 2, 2.5, 5, 10].map((m) => m * pow).find((s) => s >= raw);
  }

  function ticks(lo, hi, target) {
    const step = niceStep(hi - lo, target);
    const out = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Math.round(t / step) * step);
    return { values: out, places: Math.max(0, -Math.floor(Math.log10(step) + 1e-9)) };
  }

  function drawMark(group, s, x, y, cls, parent, scale = 1) {
    if (s.shape === "square") {
      const h = s.r * scale;
      return sv("rect", { x: x - h, y: y - h, width: 2 * h, height: 2 * h, rx: 2, class: cls }, parent);
    }
    return sv("circle", { cx: x, cy: y, r: s.r * scale, class: cls }, parent);
  }

  function renderChart() {
    const box = document.getElementById("chart");
    const tooltip = document.getElementById("tooltip");
    const width = Math.round(box.clientWidth);
    if (!width) return;
    lastWidth = width;
    const narrow = width < 480;
    const height = Math.round(Math.min(460, Math.max(320, width * 0.66)));
    const m = { top: 40, right: 12, bottom: 52, left: 52 };
    const table = D.frontier[split];
    const groups = table.groups;

    const both = Object.values(D.frontier).flatMap((t) => Object.entries(t.groups))
      .filter(([k]) => !STYLE[k].offscale).flatMap(([, g]) => [g.mean, ...Object.values(g.runs)]);
    const allRuns = Object.values(D.frontier).flatMap((t) => Object.values(t.groups))
      .flatMap((g) => [g.mean, ...Object.values(g.runs)]);
    // One frame for both splits, so switching moves the points, not the axes.
    const xs = allRuns.map((p) => p.general), ys = both.map((p) => p.personal);
    const padX = 0.06 * (Math.max(...xs) - Math.min(...xs)), padY = 0.08 * (Math.max(...ys) - Math.min(...ys));
    const x0 = Math.min(...xs) - padX, x1 = Math.max(...xs) + padX;
    const y0 = Math.min(...ys) - padY, y1 = Math.max(...ys) + padY;
    const X = (v) => m.left + ((v - x0) / (x1 - x0)) * (width - m.left - m.right);
    const Y = (v) => m.top + ((y1 - v) / (y1 - y0)) * (height - m.top - m.bottom);

    box.querySelector("svg")?.remove();
    const svg = sv("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img",
      "aria-label": `Personal CE against general-weighted CE, ${split === "test" ? "test" : "validation"} split. The table view below has every value.` });
    box.insertBefore(svg, tooltip);

    const grid = sv("g", { class: "grid" }, svg);
    const tx = ticks(x0, x1, narrow ? 4 : 7), ty = ticks(y0, y1, 5);
    for (const t of tx.values) {
      sv("line", { x1: X(t), x2: X(t), y1: m.top, y2: height - m.bottom }, grid);
      sv("text", { x: X(t), y: height - m.bottom + 18, "text-anchor": "middle", class: "tick" }, svg).textContent = t.toFixed(tx.places);
    }
    for (const t of ty.values) {
      sv("line", { x1: m.left, x2: width - m.right, y1: Y(t), y2: Y(t) }, grid);
      sv("text", { x: m.left - 8, y: Y(t) + 4, "text-anchor": "end", class: "tick" }, svg).textContent = t.toFixed(ty.places);
    }
    sv("line", { x1: m.left, x2: width - m.right, y1: height - m.bottom, y2: height - m.bottom, class: "axis-line" }, svg);
    sv("text", { x: m.left + (width - m.left - m.right) / 2, y: height - 10, "text-anchor": "middle", class: "axis-title" }, svg)
      .textContent = narrow ? "General-weighted CE (lower is better)" : "General-weighted CE: web, dialogue, code (lower is better)";
    sv("text", { x: 14, y: m.top + (height - m.top - m.bottom) / 2, "text-anchor": "middle", class: "axis-title",
      transform: `rotate(-90 14 ${m.top + (height - m.top - m.bottom) / 2})` }, svg)
      .textContent = narrow ? "Personal CE (lower is better)" : "Personal CE: the owner's messages (lower is better)";
    sv("text", { x: m.left + 8, y: height - m.bottom - 8, class: "hint" }, svg).textContent = "↙ better on both";

    // Single runs, then the frontier, then the means and their labels.
    const points = [];
    const runLayer = sv("g", {}, svg);
    for (const k of ORDER) {
      const s = STYLE[k], g = groups[k];
      if (s.offscale || Object.keys(g.runs).length < 2) continue;
      for (const [run, p] of Object.entries(g.runs)) {
        const cx = X(p.general), cy = Y(p.personal);
        sv("circle", { cx, cy, r: 3, class: `run ${s.cls}` }, runLayer);
        points.push({ x: cx, y: cy, kind: "run", group: k, run, p });
      }
    }
    const front = table.front.map((k) => groups[k].mean);
    sv("polyline", { points: front.map((p) => `${X(p.general)},${Y(p.personal)}`).join(" "), class: "frontier" }, svg);

    const labels = sv("g", {}, svg);
    for (const k of ORDER) {
      const s = STYLE[k], g = groups[k];
      const cx = X(g.mean.general);
      if (s.offscale) {
        const cy = m.top - 22;
        sv("path", { d: `M${cx} ${cy + 12} V${cy - 2} M${cx - 4} ${cy + 2} L${cx} ${cy - 3} L${cx + 4} ${cy + 2}`, class: "offscale-arrow" }, svg);
        const t = sv("text", { x: cx + 9, y: cy + 8, class: "label" }, labels);
        t.textContent = `${s.label} `;
        sv("tspan", { class: "sub" }, t).textContent = narrow ? FORMATS.f2(g.mean.personal) : `personal ${FORMATS.f2(g.mean.personal)}, off scale: never saw the chat`;
        points.push({ x: cx, y: cy + 4, kind: "mean", group: k, p: g.mean, main: true });
        continue;
      }
      const cy = Y(g.mean.personal);
      drawMark(k, s, cx, cy, `mean ${s.cls}`, svg);
      const [dx, dy, anchor] = s.place;
      const t = sv("text", { x: cx + dx, y: cy + dy, "text-anchor": anchor, class: "label" }, labels);
      t.textContent = s.label;
      if (s.sub && !narrow) sv("tspan", { class: "sub" }, t).textContent = ` · ${s.sub}`;
      points.push({ x: cx, y: cy, kind: "mean", group: k, p: g.mean, main: true });
    }
    // Keep every label inside the frame.
    for (const t of labels.querySelectorAll("text")) {
      const b = t.getBBox();
      if (b.x + b.width > width - 2) { t.setAttribute("x", width - 2); t.setAttribute("text-anchor", "end"); }
      if (b.x < m.left + 2) { t.setAttribute("x", m.left + 2); t.setAttribute("text-anchor", "start"); }
    }

    // Hover and focus: the nearest point within reach, means first.
    const ring = sv("circle", { r: 0, class: "ring", visibility: "hidden" }, svg);
    const show = (pt) => {
      if (!pt) { tooltip.hidden = true; ring.setAttribute("visibility", "hidden"); return; }
      const g = groups[pt.group], s = STYLE[pt.group];
      tooltip.replaceChildren();
      el("div", { class: "t-title" }, tooltip, pt.kind === "run" ? `${g.label} · one seed` : g.label);
      el("div", { class: "t-sub" }, tooltip, `${g.size} · ${g.what}`);
      for (const [v, name] of [[pt.p.personal, "personal CE"], [pt.p.general, "general-weighted CE"]]) {
        const row = el("div", { class: "t-row" }, tooltip);
        el("strong", {}, row, FORMATS.f3(v));
        el("span", {}, row, name);
      }
      const n = Object.keys(g.runs).length;
      el("div", { class: "t-foot" }, tooltip, pt.kind === "run" ? `run ${pt.run}` : n > 1 ? `mean of ${n} seeds` : "a single run");
      tooltip.hidden = false;
      const left = Math.min(Math.max(pt.x + 14, 0), width - tooltip.offsetWidth);
      const top = pt.y - tooltip.offsetHeight - 12 < 0 ? pt.y + 14 : pt.y - tooltip.offsetHeight - 12;
      tooltip.style.left = `${pt.kind === "mean" && pt.x + 14 + tooltip.offsetWidth > width ? pt.x - tooltip.offsetWidth - 14 : left}px`;
      tooltip.style.top = `${top}px`;
      ring.setAttribute("cx", pt.x); ring.setAttribute("cy", pt.y);
      ring.setAttribute("r", pt.kind === "run" ? 6 : (s.r || 5) + 4);
      ring.setAttribute("visibility", "visible");
    };
    const nearest = (mx, my) => {
      let best = null, bestD = Infinity;
      for (const pt of points) {
        const d = Math.hypot(pt.x - mx, pt.y - my) - (pt.kind === "mean" ? 6 : 0);
        if (d < bestD) { best = pt; bestD = d; }
      }
      return bestD <= 18 ? best : null;
    };
    svg.addEventListener("pointermove", (e) => {
      const r = svg.getBoundingClientRect();
      show(nearest((e.clientX - r.left) * (width / r.width), (e.clientY - r.top) * (height / r.height)));
    });
    svg.addEventListener("pointerleave", () => show(null));
    for (const pt of points.filter((p) => p.main)) {
      const hit = sv("circle", { cx: pt.x, cy: pt.y, r: 14, class: "hit", tabindex: 0,
        "aria-label": `${groups[pt.group].label}: personal CE ${FORMATS.f3(pt.p.personal)}, general-weighted CE ${FORMATS.f3(pt.p.general)}` }, svg);
      hit.addEventListener("focus", () => show(pt));
      hit.addEventListener("blur", () => show(null));
    }

    renderFrontierTable(table);
  }

  function renderFrontierTable(table) {
    const body = document.querySelector("#frontier-table tbody");
    body.replaceChildren();
    for (const k of ORDER) {
      const g = table.groups[k], s = STYLE[k];
      const tr = el("tr", {}, body);
      const name = el("td", {}, tr);
      el("span", { class: `swatch ${s.cls}`, "aria-hidden": "true" }, name);
      name.appendChild(document.createTextNode(g.label));
      el("td", {}, tr, g.size);
      el("td", {}, tr, String(Object.keys(g.runs).length));
      el("td", {}, tr, FORMATS.f3(g.mean.personal));
      el("td", {}, tr, FORMATS.f3(g.mean.general));
      el("td", {}, tr, table.front.includes(k) ? "yes" : "");
    }
  }

  function setupSplit() {
    const buttons = document.querySelectorAll("[data-split]");
    for (const b of buttons) {
      b.addEventListener("click", () => {
        split = b.dataset.split;
        for (const o of buttons) o.setAttribute("aria-pressed", String(o === b));
        renderChart();
      });
    }
    const box = document.getElementById("chart");
    new ResizeObserver(() => { if (Math.round(box.clientWidth) !== lastWidth) renderChart(); }).observe(box);
  }

  // --- tables ------------------------------------------------------------

  function renderFinalTable() {
    const body = document.querySelector("#final-table tbody");
    let section = null;
    for (const r of D.final.rows) {
      const kind = r.primary ? "Primary: the rule's two axes" : "Exploratory: reported, never judged";
      if (kind !== section) {
        section = kind;
        el("td", { colspan: 5 }, el("tr", { class: "section" }, body), kind);
      }
      const tr = el("tr", r.primary ? { class: "highlight" } : {}, body);
      const name = el("td", {}, tr, r.metric);
      el("span", { class: "better" }, name, `${r.better} is better`);
      // The primary axes to four places, as README's tables give them; the rest to three.
      const [f, s] = r.primary ? [FORMATS.f4, FORMATS.s4] : [FORMATS.f3, FORMATS.s3];
      el("td", {}, tr, f(r.flagship_test));
      el("td", {}, tr, f(r.final_test));
      el("td", {}, tr, s(r.difference_test));
      el("td", {}, tr, s(r.difference_val));
    }
  }

  function renderMemoTable() {
    const body = document.querySelector("#memo-table tbody");
    const M = D.memorisation;
    const names = { "flagship + C": "Flagship + C, the final model", flagship: "Flagship" };
    for (const g of M.groups) {
      const tr = el("tr", g.group === "flagship + C" ? { class: "highlight" } : {}, body);
      el("td", {}, tr, `${names[g.group] || g.group}${g.n > 1 ? `, ${g.n} seeds` : ""}`);
      el("td", {}, tr, FORMATS.pct1(g.train_rate));
      el("td", {}, tr, FORMATS.pct1(g.val_rate));
      el("td", {}, tr, FORMATS.f2(g.overlap_mean));
      el("td", {}, tr, FORMATS.f1(g.overlap_p95));
    }
    const tr = el("tr", {}, body);
    el("td", {}, tr, "The owner’s real held-out messages");
    el("td", {}, tr, "\u2014");
    el("td", {}, tr, "\u2014");
    el("td", {}, tr, FORMATS.f2(M.real.overlap_mean));
    el("td", {}, tr, FORMATS.f1(M.real.overlap_p95));
  }

  // --- samples -----------------------------------------------------------

  const MODELS = { final: ["Flagship + C, the final model", "final"], flagship: ["Flagship, unadapted", "flagship"] };
  const FAMILIES = { voice: "voice: the next message", degeneration: "continuing general text", code: "continuing a function" };

  async function loadSamples() {
    const url = document.querySelector('meta[name="minime:samples"]').content.trim();
    const list = document.getElementById("samples-list");
    if (!url) return;
    let payload;
    try {
      const res = await fetch(url, { cache: "no-cache", credentials: "omit" });
      // S3 answers 403 for a missing key the policy names: taken down, or not up yet.
      if (res.status === 403 || res.status === 404) payload = { samples: [] };
      else if (!res.ok) throw new Error(String(res.status));
      else payload = await res.json();
    } catch {
      list.replaceChildren();
      el("p", { class: "empty" }, list, "The samples could not be loaded.");
      return;
    }
    const samples = (payload && Array.isArray(payload.samples)) ? payload.samples : [];
    list.replaceChildren();
    if (!samples.length) { el("p", { class: "empty" }, list, "The samples are not published yet."); return; }
    let redacted = false;
    for (const s of samples) {
      const [model, cls] = MODELS[s.model] || [String(s.model), "probe"];
      const card = el("article", { class: "card sample" }, list);
      const meta = el("div", { class: "meta" }, card);
      const who = el("span", { class: "model" }, meta);
      el("span", { class: `swatch ${cls}`, "aria-hidden": "true" }, who);
      who.appendChild(document.createTextNode(model));
      el("span", {}, meta, FAMILIES[s.family] || String(s.family));
      el("span", {}, meta, s.split === "test" ? "test split" : "validation split");
      const text = String(s.text);
      el("p", { class: s.family === "code" ? "text code" : "text" }, card, text);
      if (/<(NAME|HANDLE|PHONE|EMAIL|URL|NUMBER)>/.test(text)) redacted = true;
    }
    document.getElementById("redaction-note").hidden = !redacted;
  }

  function linkRepository() {
    const url = document.querySelector('meta[name="minime:repository"]').content.trim();
    if (!/^https:\/\/github\.com\//.test(url)) return;
    const a = document.getElementById("repo-link");
    a.href = url;
    a.textContent = url.replace(/^https:\/\//, "");
    document.getElementById("repo-line").hidden = false;
  }

  function start() {
    if (!D) return;
    bind();
    renderFinalTable();
    renderMemoTable();
    setupSplit();
    renderChart();
    linkRepository();
    loadSamples();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
