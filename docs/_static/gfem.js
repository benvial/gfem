// gfem additions to Shibuya's own JS. Shibuya toggles the drawer (#head-nav) and
// sets aria-expanded; this adds the rest of the gfem behaviour.
(function () {
  // 1. Header nav: mark the current section (Shibuya's nav-links.html emits no current state).
  const here = location.pathname;
  const links = [...document.querySelectorAll(".sy-head-links a[href]")];
  const section = (a) => new URL(a.getAttribute("href"), location.href).pathname.replace(/[^/]*$/, "");
  const brand = document.querySelector(".sy-head-brand");
  const root = brand ? new URL(brand.getAttribute("href"), location.href).pathname.replace(/[^/]*$/, "") : "/";
  let best = null;
  for (const a of links) {
    const p = new URL(a.getAttribute("href"), location.href).pathname;
    const s = section(a);
    if (p === here || (s !== root && here.startsWith(s) && (!best || s.length > section(best).length))) best = a;
  }
  // a section is a directory (tutorials/, recipes/, api/); single pages match exactly
  if (best) best.setAttribute("aria-current", "page");

  // 2. Notebook cells cannot take :emphasize-lines:, so a cell tagged "decisive" gets its
  //    jax.grad / jax.value_and_grad line washed like an emphasized code-block line.
  document.querySelectorAll(".cell.tag_decisive .cell_input .highlight pre > span[data-line]").forEach((l) => {
    if (/jax\.(value_and_)?grad\(/.test(l.textContent)) l.innerHTML = '<span class="hll">' + l.innerHTML + "</span>";
  });
  // 3. On-page TOC: before the first scroll, mark the first section (Shibuya's scroll-spy takes over).
  const toc = document.querySelector(".localtoc li");
  if (toc && !document.querySelector(".localtoc li.active") && scrollY < 40) toc.classList.add("active");

  // 4. Search modal: the "search page" link carries the typed query to Sphinx's search page.
  document.querySelector(".gfem-search-page")?.addEventListener("click", (e) => {
    const q = document.querySelector("#gfem-search-modal .pf-input")?.value.trim();
    if (q) e.currentTarget.search = "?q=" + encodeURIComponent(q);
  });

  // 5. Drawer: focus moves to the first link on open; Escape closes and returns focus to Menu.
  const btn = document.querySelector(".gfem-menu"), nav = document.getElementById("head-nav");
  if (!btn || !nav) return;
  btn.addEventListener("click", () => {
    requestAnimationFrame(() => {
      if (btn.getAttribute("aria-expanded") === "true") nav.querySelector(".sy-head-links a, .gfem-drawer-tree a")?.focus();
    });
  });
  addEventListener("keydown", (e) => {
    if (e.key === "Escape" && btn.getAttribute("aria-expanded") === "true") { btn.click(); btn.focus(); }
  });
})();
