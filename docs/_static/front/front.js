// gfem front page (layout/gfem-home.html): draws the hero's two field maps and the
// four design snapshots from window.BEND, which conf.py inlines from
// docs/_data/front-page.json (tools/front_page_data.py, the 3D bend's run record).
// Shibuya owns the theme switch (html.light / html.dark) and the menu; the figures
// redraw, without the fade-in, when the theme class or the width changes.
(function () {
  const ns = "http://www.w3.org/2000/svg";
  const el = (name, attrs) => {
    const node = document.createElementNS(ns, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  };
  const bytes = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

  // A raster (uint8, rows from max y down) as a data URL: rgb at alpha(v), v in [0, 1].
  function raster(values, [ny, nx], rgb, alpha) {
    const canvas = document.createElement("canvas");
    canvas.width = nx;
    canvas.height = ny;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(nx, ny);
    const [r, g, b] = rgb.split(",").map(Number);
    values.forEach((v, i) => img.data.set([r, g, b, Math.round(255 * alpha(v / 255))], 4 * i));
    ctx.putImageData(img, 0, 0);
    return canvas.toDataURL();
  }

  // An SVG in um, y up: viewBox over the extent [x0, x1, y0, y1], drawn at its width.
  function frame(svgEl, [x0, x1, y0, y1]) {
    const W = svgEl.clientWidth;
    svgEl.setAttribute("viewBox", `${x0} ${-y1} ${x1 - x0} ${y1 - y0}`);
    svgEl.style.height = (W * (y1 - y0)) / (x1 - x0) + "px";
    svgEl.replaceChildren();
  }

  const line = (pts) => "M" + pts.map(([x, y]) => `${x},${-y}`).join("L");
  const css = () => getComputedStyle(document.documentElement);

  // |E| on the core mid-plane: the field in the signal red, the silicon edge in ink,
  // the design square dashed.
  function drawField(svgEl, data, name, animate) {
    const ext = data.extent_um, d = data.design_um, style = css();
    frame(svgEl, ext);
    const image = el("image", {
      href: raster(bytes(data.field[name]), data.field.shape, style.getPropertyValue("--accent-field").trim(),
        (v) => Math.min(0.92, 0.95 * v)),
      x: ext[0], y: -ext[3], width: ext[1] - ext[0], height: ext[3] - ext[2], preserveAspectRatio: "none",
    });
    svgEl.appendChild(image);
    svgEl.appendChild(el("rect", { x: -d, y: -d, width: 2 * d, height: 2 * d, class: "square" }));
    for (const pts of data.edge[name]) svgEl.appendChild(el("path", { d: line(pts), class: "edge" }));
    if (animate && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
      image.style.opacity = 0;
      image.style.transition = "opacity .9s cubic-bezier(.16,1,.3,1)";
      requestAnimationFrame(() => requestAnimationFrame(() => (image.style.opacity = 1)));
    }
  }

  // A design snapshot over the whole box: the guides, then the design square's density in ink.
  function drawDesign(svgEl, data, k) {
    const ext = data.extent_um, d = data.design_um, w = data.guide_um, style = css();
    const ink = style.getPropertyValue("--ink-field").trim();
    frame(svgEl, ext);
    const fill = `rgba(${ink},.88)`;
    svgEl.appendChild(el("rect", { x: ext[0], y: -w, width: ext[0] * -1 - d, height: 2 * w, fill }));
    svgEl.appendChild(el("rect", { x: -w, y: d, width: 2 * w, height: -ext[2] - d, fill }));
    svgEl.appendChild(el("image", {
      href: raster(bytes(data.snapshots.density[k]), data.snapshots.shape, ink, (v) => 0.88 * v),
      x: -d, y: -d, width: 2 * d, height: 2 * d, preserveAspectRatio: "none", class: "pixels",
    }));
    svgEl.appendChild(el("rect", { x: -d, y: -d, width: 2 * d, height: 2 * d, class: "square" }));
  }

  // Pointer move or tap over a field shows |E| / input-guide peak there.
  function readout(svgEl, tip, data, name) {
    const values = bytes(data.field[name]);
    const [ny, nx] = data.field.shape;
    const show = (e) => {
      const r = svgEl.getBoundingClientRect(), box = tip.parentElement.getBoundingClientRect();
      const col = Math.floor(((e.clientX - r.left) / r.width) * nx);
      const row = Math.floor(((e.clientY - r.top) / r.height) * ny);
      if (col < 0 || col >= nx || row < 0 || row >= ny) { tip.hidden = true; return; }
      tip.hidden = false;
      tip.textContent = `|E| / input peak = ${((values[row * nx + col] / 255) * data.field.vmax).toFixed(2)}`;
      const x = e.clientX - box.left, flip = x > box.width - 200;
      tip.style.left = (flip ? x - 190 : x + 14) + "px";
      tip.style.top = e.clientY - box.top + 14 + "px";
    };
    svgEl.addEventListener("pointermove", show);
    svgEl.addEventListener("pointerdown", show);
    svgEl.addEventListener("pointerleave", () => (tip.hidden = true));
  }

  function drawFront(data, first) {
    for (const name of ["arc", "final"]) {
      const svgEl = document.getElementById("f-" + name);
      drawField(svgEl, data, name, first && name === "final");
      if (first) readout(svgEl, document.getElementById("grad-tip"), data, name);
    }
    data.snapshots.density.forEach((_, k) => drawDesign(document.getElementById("s-" + k), data, k));
  }

  if (window.BEND) {
    let first = true;
    const draw = () => { drawFront(window.BEND, first); first = false; };
    draw();
    // redraw on a theme switch only: html's class list changes for other reasons too
    const root = document.documentElement;
    let dark = root.classList.contains("dark");
    new MutationObserver(() => {
      if (root.classList.contains("dark") === dark) return;
      dark = !dark;
      draw();
    }).observe(root, { attributes: true, attributeFilter: ["class"] });
    let resizeTimer;
    addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(draw, 150); });
  }

  // The install line's copy button. Without clipboard access (an insecure context, a
  // denied permission) it selects the line instead, for a manual copy.
  const copy = document.querySelector(".gfem-copy");
  copy?.addEventListener("click", () => {
    const pre = copy.parentElement;
    const text = [...pre.childNodes].filter((n) => n !== copy).map((n) => n.textContent).join("").trim();
    const flash = (label) => {
      copy.textContent = label;
      setTimeout(() => (copy.textContent = "copy"), 1500);
    };
    const selectLine = () => {
      const range = document.createRange();
      range.selectNodeContents(pre);
      range.setStartAfter(copy);
      getSelection().removeAllRanges();
      getSelection().addRange(range);
      flash("selected");
    };
    if (!navigator.clipboard) return selectLine();
    navigator.clipboard.writeText(text).then(() => flash("copied"), selectLine);
  });
})();
