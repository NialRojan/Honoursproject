const fixtureSelect = document.getElementById("fixture-select");
const urlInput = document.getElementById("url-input");
const screenshot = document.getElementById("screenshot");
const overlay = document.getElementById("overlay");
const findingsList = document.getElementById("findings-list");
const riskBadge = document.getElementById("risk-badge");
const riskScoreValue = document.getElementById("risk-score-value");

const ENGINE_COLORS = {
  structure: "#e74c3c",
  visual: "#3498db",
  linguistic: "#9b59b6",
  behavior: "#f39c12",
};

async function loadFixtures() {
  const res = await fetch("/fixtures");
  const data = await res.json();
  fixtureSelect.innerHTML = data.fixtures.map(f => `<option value="${f}">${f}</option>`).join("");
}

function elementBySelector(elements, selector) {
  return elements.find(e => e.selector === selector);
}

function renderEngineErrors(engineErrors) {
  if (!engineErrors) return;
  Object.entries(engineErrors).forEach(([engine, message]) => {
    const li = document.createElement("li");
    li.className = "engine-error";
    li.innerHTML = `<strong>${engine} engine error</strong><br><small>${message}</small>`;
    findingsList.appendChild(li);
  });
}

function drawBoxes(data) {
  overlay.innerHTML = "";

  // The screenshot's natural pixel size (e.g. 1280x720) is scaled down by
  // the browser to fit #screenshot's CSS width (100% of its container), but
  // the API returns box coordinates in raw, unscaled screenshot pixels. We
  // must scale every box by the same factor the browser scaled the image,
  // or boxes land in the wrong place. Aspect ratio is preserved (only CSS
  // width is constrained), so a single uniform scale factor derived from
  // width is correct for both axes.
  const scale = screenshot.naturalWidth
    ? screenshot.clientWidth / screenshot.naturalWidth
    : 1;

  data.findings.forEach((f) => {
    if (!f.element_ref) return;
    const el = elementBySelector(data.elements, f.element_ref);
    if (!el) return;
    const [x1, y1, x2, y2] = el.bbox;
    const box = document.createElement("div");
    box.className = "finding-box";
    box.style.left = (x1 * scale) + "px";
    box.style.top = (y1 * scale) + "px";
    box.style.width = ((x2 - x1) * scale) + "px";
    box.style.height = ((y2 - y1) * scale) + "px";
    box.style.borderColor = ENGINE_COLORS[f.engine] || "#999";
    overlay.appendChild(box);
  });
}

function render(data) {
  riskBadge.classList.remove("hidden");
  riskScoreValue.textContent = data.risk_score;

  overlay.innerHTML = "";
  findingsList.innerHTML = "";

  renderEngineErrors(data.engine_errors);

  data.findings.forEach((f, i) => {
    const li = document.createElement("li");
    li.style.borderLeftColor = ENGINE_COLORS[f.engine] || "#999";
    li.innerHTML = `<strong>${f.engine}</strong>: ${f.pattern_type} (confidence ${f.confidence})<br><small>${f.evidence}</small>`;
    findingsList.appendChild(li);
  });

  // Boxes depend on the screenshot's *rendered* size, which is only known
  // once the browser has laid out the image. Setting .src is async, so we
  // draw boxes in the onload handler (and also handle the case where the
  // image is already cached/loaded, in which case onload may not fire).
  screenshot.onload = () => drawBoxes(data);
  screenshot.src = "data:image/png;base64," + data.screenshot_base64;
  screenshot.style.display = "block";
  if (screenshot.complete && screenshot.naturalWidth) {
    drawBoxes(data);
  }
}

async function analyze(payload) {
  findingsList.innerHTML = "<li>Analyzing...</li>";
  const res = await fetch("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    findingsList.innerHTML = `<li>Error: ${res.status}</li>`;
    return;
  }
  render(await res.json());
}

document.getElementById("analyze-url-btn").addEventListener("click", () => {
  if (urlInput.value.trim()) analyze({ url: urlInput.value.trim() });
});

document.getElementById("analyze-fixture-btn").addEventListener("click", () => {
  analyze({ fixture_id: fixtureSelect.value });
});

loadFixtures();
