// gfem front page (layout/gfem-home.html): plays the hero's optimization video with
// its readout and history strip, and draws the four design snapshots, from
// window.BEND, which conf.py inlines from docs/_data/front-page.json
// (tools/front_page_data.py, the 3D bend's run record) and the video's timing.
// Shibuya owns the theme switch (html.light / html.dark) and the menu; the figures
// redraw (the video swaps to the theme's file) when the theme class or the width changes.
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

  const css = () => getComputedStyle(document.documentElement);

  // A design over the whole box: the guides, then the design square's density
  // (base64 uint8, data.snapshots.shape) in ink.
  function drawDesign(svgEl, data, density) {
    const ext = data.extent_um, d = data.design_um, w = data.guide_um, style = css();
    const ink = style.getPropertyValue("--ink-field").trim();
    frame(svgEl, ext);
    const fill = `rgba(${ink},.88)`;
    svgEl.appendChild(el("rect", { x: ext[0], y: -w, width: ext[0] * -1 - d, height: 2 * w, fill }));
    svgEl.appendChild(el("rect", { x: -w, y: d, width: 2 * w, height: -ext[2] - d, fill }));
    svgEl.appendChild(el("image", {
      href: raster(bytes(density), data.snapshots.shape, ink, (v) => 0.88 * v),
      x: -d, y: -d, width: 2 * d, height: 2 * d, preserveAspectRatio: "none", class: "pixels",
    }));
    svgEl.appendChild(el("rect", { x: -d, y: -d, width: 2 * d, height: 2 * d, class: "square" }));
  }

  // The optimization video (tools/bend3d_animation.py): the theme's file, the
  // readout of the iteration on screen and the |S21|^2 history strip's playhead.
  function film(data) {
    const video = document.getElementById("bend-video");
    if (!video) return { redraw() {} };
    const timing = data.video, its = timing.iterations, n = its.length;
    const hist = document.getElementById("hist"), button = document.getElementById("r-play");
    const out = { k: document.getElementById("r-k"), beta: document.getElementById("r-beta"), t: document.getElementById("r-t") };
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const ext = video.canPlayType('video/webm; codecs="vp9"') ? ".webm" : ".mp4";
    let theme = null, marker = null;

    const at = (time) => {  // the iteration on screen at ``time``: 0..n-1, or n for final
      if (time >= timing.final.time_s) return n;
      let k = 0;
      while (k + 1 < n && its[k + 1].time_s <= time) k++;
      return k;
    };
    const show = () => {
      const time = video.currentTime, k = at(time);
      out.k.textContent = k === n ? "final" : k;
      out.beta.textContent = its[Math.min(k, n - 1)].beta;
      out.t.textContent = (k === n ? timing.final.t : its[k].t).toFixed(4);
      if (marker) {
        const blend = its[1] ? its[1].time_s : 1;
        const x = Math.min(n - 1, time / blend);
        marker.setAttribute("transform", `translate(${marker.scale * x},0)`);
      }
    };
    const loop = () => {
      show();
      if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(loop);
      else if (!video.paused) requestAnimationFrame(loop);
    };
    const label = () => (button.textContent = video.paused ? "play" : "pause");

    function load() {
      const next = document.documentElement.classList.contains("dark") ? "dark" : "light";
      if (next === theme) return;
      const first = theme === null;
      const time = video.currentTime, paused = first ? reduce : video.paused;
      theme = next;
      video.poster = video.dataset[theme] + ".png";
      video.src = video.dataset[theme] + ext;
      video.addEventListener("loadedmetadata", () => {
        // first load with reduced motion: hold the last frame, final, paused;
        // a theme switch resumes where the other theme's file was
        video.currentTime = first ? (reduce ? video.duration - 0.05 : 0) : time;
        if (!paused) video.play().catch(label);
        label();
        show();
      }, { once: true });
    }

    function strip() {
      const W = hist.clientWidth, H = 56, pad = 6;
      hist.setAttribute("viewBox", `0 0 ${W} ${H}`);
      hist.style.height = H + "px";
      hist.replaceChildren();
      const x = (k) => pad + ((W - 2 * pad) * k) / (n - 1);
      const y = (t) => H - pad - (H - 2 * pad) * t;
      for (let k = 1; k < n; k++) {
        if (its[k].beta !== its[k - 1].beta) {
          hist.appendChild(el("line", { x1: x(k - 0.5), x2: x(k - 0.5), y1: pad, y2: H - pad, class: "tick" }));
        }
      }
      hist.appendChild(el("path", { d: "M" + its.map((it, k) => `${x(k).toFixed(1)},${y(it.t).toFixed(1)}`).join("L"), class: "curve" }));
      marker = el("g", { class: "head" });
      marker.scale = (W - 2 * pad) / (n - 1);
      marker.appendChild(el("line", { x1: pad, x2: pad, y1: pad - 4, y2: H - pad + 4 }));
      hist.appendChild(marker);
      show();
    }

    hist.addEventListener("click", (e) => {
      const r = hist.getBoundingClientRect();
      const k = Math.round(((e.clientX - r.left - 6) / (r.width - 12)) * (n - 1));
      video.currentTime = its[Math.max(0, Math.min(n - 1, k))].time_s;
      show();
    });
    button.addEventListener("click", () => {
      if (video.paused) video.play().catch(label);
      else video.pause();
    });
    video.addEventListener("play", () => { label(); loop(); });
    video.addEventListener("pause", label);
    video.addEventListener("seeked", show);
    load();
    strip();
    return { redraw() { load(); strip(); } };
  }

  function drawFront(data) {
    data.snapshots.density.forEach((d, k) => drawDesign(document.getElementById("s-" + k), data, d));
  }

  if (window.BEND) {
    const movie = film(window.BEND);
    const draw = () => { drawFront(window.BEND); movie.redraw(); };
    drawFront(window.BEND);
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
