// gfem front page (layout/gfem-home.html): draws the hero and the three etch
// variants from window.TOPO, which conf.py inlines from docs/_generated/front-page.json
// (tools/front_page_data.py, the topology tutorial's own solve).
// Shibuya owns the theme switch (html.light / html.dark) and the menu; the figures
// redraw, without the converge reveal, when the theme class or the width changes.
(function () {
  const ns = "http://www.w3.org/2000/svg";

  // Draw the design layer's top-face mesh into svgEl; value(i) is in [0, 1].
  // mode "sens": the teal sensitivity ramp (alpha = min(.92, .95 v)); "design": ink density.
  // halves: the stacked two-row render (x split at the layer's middle) for narrow screens.
  function drawLayer(svgEl, data, value, opts) {
    const [x0, x1, y0, y1] = data.extent_mm;
    const W = svgEl.clientWidth;
    const halves = opts.halves;
    const span = halves ? (x1 - x0) / 2 : x1 - x0, depth = y1 - y0;
    const gap = halves ? 10 : 0;
    const s = W / span, rowH = depth * s, H = halves ? 2 * rowH + gap : rowH;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.style.height = H + "px";
    svgEl.replaceChildren();
    const c = getComputedStyle(document.documentElement);
    const field = c.getPropertyValue("--accent-field").trim();
    const inkRGB = c.getPropertyValue("--ink-field").trim();
    const mesh = c.getPropertyValue("--mesh").trim();
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const P = data.nodes;
    data.tris.forEach((t, i) => {
      // keep a triangle whole in its row: its centroid picks the half
      const cx = (P[t[0]][0] + P[t[1]][0] + P[t[2]][0]) / 3;
      const shift = halves && cx > x0 + span ? span : 0;
      const pts = t.map((k) => {
        const x = (P[k][0] - x0 - shift) * s, y = (y1 - P[k][1]) * s + (shift ? rowH + gap : 0);
        return x.toFixed(1) + "," + y.toFixed(1);
      });
      const v = value(i);
      const path = document.createElementNS(ns, "path");
      path.setAttribute("d", "M" + pts.join("L") + "Z");
      path.setAttribute("fill", opts.mode === "sens"
        ? `rgba(${field},${Math.min(0.92, 0.95 * v).toFixed(3)})`
        : `rgba(${inkRGB},${(0.88 * v).toFixed(3)})`);
      path.setAttribute("stroke", mesh);
      path.setAttribute("stroke-width", "0.6");
      path.dataset.v = v.toFixed(3);
      if (opts.animate && !reduce) {
        // the converge reveal: high-sensitivity elements resolve first
        path.style.opacity = 0;
        path.style.transition = `opacity .9s cubic-bezier(.16,1,.3,1) ${(1 - v) * 700}ms`;
        requestAnimationFrame(() => requestAnimationFrame(() => (path.style.opacity = 1)));
      }
      svgEl.appendChild(path);
    });
  }

  // Pointer move or tap outlines an element and shows its value.
  function readout(svgEl, tip, label) {
    const clear = () => svgEl.querySelectorAll(".hot").forEach((n) => n.classList.remove("hot"));
    const show = (e) => {
      const el = e.target.closest("path");
      clear();
      if (!el) { tip.hidden = true; return; }
      el.classList.add("hot");
      tip.hidden = false;
      tip.textContent = `${label} = ${el.dataset.v}`;
      const r = svgEl.getBoundingClientRect();
      const x = e.clientX - r.left, flip = x > r.width - 160;
      tip.style.left = (flip ? x - 150 : x + 14) + "px";
      tip.style.top = e.clientY - r.top + 14 + "px";
    };
    svgEl.addEventListener("pointermove", show);
    svgEl.addEventListener("pointerdown", show);
    svgEl.addEventListener("pointerleave", () => { tip.hidden = true; clear(); });
  }

  function drawFront(data, first) {
    const hero = document.getElementById("grad");
    drawLayer(hero, data, (i) => data.sens0[i], { mode: "sens", animate: first, halves: innerWidth < 600 });
    if (first) readout(hero, document.getElementById("grad-tip"), "|∂J/∂ε_r|, normalised");
    for (const name of ["eroded", "nominal", "dilated"]) {
      drawLayer(document.getElementById("v-" + name), data, (i) => data.variants[name][i], { mode: "design", halves: false });
    }
  }

  if (window.TOPO) {
    let first = true;
    const draw = () => { drawFront(window.TOPO, first); first = false; };
    draw();
    new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    let rt;
    addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(draw, 150); });
  }

  // The install line's copy button.
  const copy = document.querySelector(".gfem-copy");
  copy?.addEventListener("click", () => {
    const text = [...copy.parentElement.childNodes].filter((n) => n !== copy).map((n) => n.textContent).join("").trim();
    navigator.clipboard?.writeText(text).then(() => {
      copy.textContent = "copied";
      setTimeout(() => (copy.textContent = "copy"), 1500);
    });
  });
})();
