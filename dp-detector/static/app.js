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

// The risk score in the header always reflects every engine's findings
// combined, regardless of the active filter below -- the filter only
// changes which findings are *shown*, not what's included in the score.
let lastData = null;
let activeEngine = "all";

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

function filteredFindings(data) {
  if (activeEngine === "all") return data.findings;
  return data.findings.filter((f) => f.engine === activeEngine);
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

  filteredFindings(data).forEach((f) => {
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

function renderFindingsList(data) {
  findingsList.innerHTML = "";

  // Engine errors are shown regardless of the active filter -- if you've
  // filtered to "behavior" and it failed to run, you should still see why,
  // not an empty list that looks like "nothing to report".
  renderEngineErrors(data.engine_errors);

  const findings = filteredFindings(data);
  if (findings.length === 0 && !(data.engine_errors && Object.keys(data.engine_errors).length)) {
    const li = document.createElement("li");
    li.textContent = activeEngine === "all"
      ? "No findings."
      : `No ${activeEngine} findings.`;
    findingsList.appendChild(li);
    return;
  }

  findings.forEach((f) => {
    const li = document.createElement("li");
    li.style.borderLeftColor = ENGINE_COLORS[f.engine] || "#999";
    li.innerHTML = `<strong>${f.engine}</strong>: ${f.pattern_type} (confidence ${f.confidence})<br><small>${f.evidence}</small>`;
    findingsList.appendChild(li);
  });
}

function setFilterButtonUI(engine) {
  document.querySelectorAll(".filter-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.engine === engine);
  });
}

// Called when the user clicks a filter button, i.e. after the screenshot
// for the current analysis is already loaded -- safe to draw boxes
// immediately, unlike the initial render() below.
function setActiveEngine(engine) {
  activeEngine = engine;
  setFilterButtonUI(engine);
  if (lastData) {
    renderFindingsList(lastData);
    drawBoxes(lastData);
  }
}

function render(data) {
  lastData = data;
  activeEngine = "all";
  setFilterButtonUI("all");

  // The risk score always reflects every engine combined -- filtering only
  // changes which findings are displayed below, never the score itself.
  riskBadge.classList.remove("hidden");
  riskScoreValue.textContent = data.risk_score;

  renderFindingsList(data);

  // Boxes depend on the screenshot's *rendered* size, which is only known
  // once the browser has laid out the image. Setting .src is async, so we
  // draw boxes in the onload handler (and also handle the case where the
  // image is already cached/loaded, in which case onload may not fire).
  screenshot.onload = () => drawBoxes(lastData);
  screenshot.src = "data:image/png;base64," + data.screenshot_base64;
  screenshot.style.display = "block";
  if (screenshot.complete && screenshot.naturalWidth) {
    drawBoxes(lastData);
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
    // Show the API's actual detail message (e.g. "url must use http or
    // https", "capture failed: ...") rather than a bare status code --
    // the code alone doesn't tell the user what to fix.
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (e) {
      // response body wasn't JSON; fall back to statusText
    }
    findingsList.innerHTML = `<li>Error ${res.status}: ${detail}</li>`;
    return;
  }
  render(await res.json());
}

// Most people type a bare domain (as in a browser address bar) without a
// scheme. The API deliberately rejects that (only http/https are allowed,
// to block file:// and other local-read vectors) rather than guessing, so
// normalize here instead of surfacing a confusing 400 for the common case.
function normalizeUrl(raw) {
  return /^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(raw) ? raw : "https://" + raw;
}

document.getElementById("analyze-url-btn").addEventListener("click", () => {
  const raw = urlInput.value.trim();
  if (raw) analyze({ url: normalizeUrl(raw) });
});

document.getElementById("analyze-fixture-btn").addEventListener("click", () => {
  analyze({ fixture_id: fixtureSelect.value });
});

document.querySelectorAll(".filter-btn").forEach((btn) => {
  btn.addEventListener("click", () => setActiveEngine(btn.dataset.engine));
});

loadFixtures();
