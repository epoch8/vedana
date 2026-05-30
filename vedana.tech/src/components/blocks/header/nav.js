const header = document.getElementById("site-header");
const toggle = document.getElementById("menu-toggle");
const nav = document.getElementById("primary-nav");
const overlay = document.getElementById("nav-overlay");

if (!header || !toggle || !nav || !overlay) {
  console.warn("Nav elements not found");
}

/* ================================
   STATE (через DOM, не через переменную)
================================ */

const isOpen = () => header.getAttribute("data-open") === "true";

/* ================================
   ACTIONS
================================ */

function openMenu() {
  header.setAttribute("data-open", "true");
  toggle.setAttribute("aria-expanded", "true");

  // 👉 блокируем скролл (иначе мобильный UX говно)
  document.body.style.overflow = "hidden";
}

function closeMenu() {
  header.setAttribute("data-open", "false");
  toggle.setAttribute("aria-expanded", "false");

  document.body.style.overflow = "";
}

function toggleMenu() {
  isOpen() ? closeMenu() : openMenu();
}

/* ================================
   EVENTS
================================ */

/* 👉 кнопка */
toggle.addEventListener("click", (e) => {
  e.stopPropagation();
  toggleMenu();
});

/* 👉 overlay */
overlay.addEventListener("click", closeMenu);

/* 👉 клик ВНЕ меню */
document.addEventListener("click", (e) => {
  if (!isOpen()) return;

  const target = e.target;

  if (
    !nav.contains(target) &&
    !toggle.contains(target)
  ) {
    closeMenu();
  }
});

/* 👉 ESC */
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && isOpen()) {
    closeMenu();
  }
});

/* 👉 защита от resize (если открыли на мобилке → десктоп) */
window.addEventListener("resize", () => {
  if (window.innerWidth > 768 && isOpen()) {
    closeMenu();
  }
});