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

function render(data) {
  riskBadge.classList.remove("hidden");
  riskScoreValue.textContent = data.risk_score;

  screenshot.src = "data:image/png;base64," + data.screenshot_base64;
  screenshot.style.display = "block";

  overlay.innerHTML = "";
  findingsList.innerHTML = "";

  data.findings.forEach((f, i) => {
    const li = document.createElement("li");
    li.style.borderLeftColor = ENGINE_COLORS[f.engine] || "#999";
    li.innerHTML = `<strong>${f.engine}</strong>: ${f.pattern_type} (confidence ${f.confidence})<br><small>${f.evidence}</small>`;
    findingsList.appendChild(li);

    if (f.element_ref) {
      const el = elementBySelector(data.elements, f.element_ref);
      if (el) {
        const [x1, y1, x2, y2] = el.bbox;
        const box = document.createElement("div");
        box.className = "finding-box";
        box.style.left = x1 + "px";
        box.style.top = y1 + "px";
        box.style.width = (x2 - x1) + "px";
        box.style.height = (y2 - y1) + "px";
        box.style.borderColor = ENGINE_COLORS[f.engine] || "#999";
        overlay.appendChild(box);
      }
    }
  });
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
