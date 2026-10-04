const tabs = [...document.querySelectorAll("[data-demo]")];
const code = document.getElementById("demo-code");
const output = document.getElementById("demo-output");
const status = document.getElementById("copy-status");
let demos;

// Highlight only source-controlled example text, escaping every token first.
function highlight(source) {
  const tokens =
    source.match(
      /#[^\n]*|"[^"\n]*"|'[^'\n]*'|\b(?:from|import|as|True|False|None|with|assert)\b|\b\d+(?:\.\d+)?\b|[^#"'\w]+|\w+|./g,
    ) || [];
  return tokens
    .map((token) => {
      const safe = token.replace(
        /[&<>"']/g,
        (char) =>
          ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;",
          })[char],
      );
      let type = "";
      if (token.startsWith("#")) type = "comment";
      else if (/^["']/.test(token)) type = "string";
      else if (/^(from|import|as|True|False|None|with|assert)$/.test(token))
        type = "keyword";
      else if (/^\d/.test(token)) type = "number";
      return type ? `<span class="syntax-${type}">${safe}</span>` : safe;
    })
    .join("");
}

function selectDemo(key, moveFocus = false) {
  const demo = demos?.[key];
  if (!demo) return;
  tabs.forEach((tab) => {
    const active = tab.dataset.demo === key;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    if (active && moveFocus) tab.focus();
  });
  document
    .getElementById("demo-panel")
    .setAttribute("aria-labelledby", `tab-${key}`);
  code.innerHTML = highlight(demo.code);
  output.innerHTML = highlight(JSON.stringify(demo.expected, null, 2));
  document.getElementById("demo-badge").textContent = demo.badge;
  document.getElementById("demo-description").textContent = demo.description;
  document.getElementById("demo-guide").href = demo.guide;
}

for (const tab of tabs) {
  tab.addEventListener("click", () => selectDemo(tab.dataset.demo));
  tab.addEventListener("keydown", (event) => {
    const index = tabs.indexOf(tab);
    let target;
    if (event.key === "ArrowRight") target = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft")
      target = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") target = 0;
    if (event.key === "End") target = tabs.length - 1;
    if (target !== undefined) {
      event.preventDefault();
      selectDemo(tabs[target].dataset.demo, true);
    }
  });
}

for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", async () => {
    const text = document.getElementById(button.dataset.copy).textContent;
    const original = button.innerHTML;
    try {
      if (!navigator.clipboard?.writeText)
        throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied ✓";
      status.textContent =
        button.dataset.copy === "install-command"
          ? "Installation command copied."
          : "Python example copied.";
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(document.getElementById(button.dataset.copy));
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = "Select & copy";
      status.textContent =
        "Clipboard unavailable. The text is selected; copy it with your keyboard or device menu.";
    }
    window.setTimeout(() => {
      button.innerHTML = original;
    }, 2200);
  });
}

code.innerHTML = highlight(code.textContent);
output.innerHTML = highlight(output.textContent);
try {
  const response = await fetch(new URL("../examples.json", import.meta.url));
  if (!response.ok) throw new Error("Examples unavailable");
  demos = await response.json();
  tabs.forEach((tab) => {
    tab.disabled = false;
  });
  selectDemo("api");
} catch {
  // Keep the complete, usable server-rendered API example if data cannot load.
  tabs.slice(1).forEach((tab) => {
    tab.disabled = true;
  });
}
