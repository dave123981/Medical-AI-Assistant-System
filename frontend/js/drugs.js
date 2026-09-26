const GATEWAY_URL = window.GATEWAY_URL || "http://localhost:8080";

const form = document.getElementById("drugs-form");
const resultEl = document.getElementById("result");
const conditionInput = document.getElementById("condition-input");
const conditionOptionsEl = document.getElementById("condition-options");
const conditionCountEl = document.getElementById("condition-count");
const ageInput = document.getElementById("age");
const medsInput = document.getElementById("current-medications");
const allergiesInput = document.getElementById("allergies");
const recommendButton = document.getElementById("recommend-button");

let allConditions = [];

function parseCommaList(value) {
  return value
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

async function loadConditions() {
  conditionCountEl.textContent = "Loading conditions…";
  recommendButton.disabled = false;

  try {
    const res = await fetch(`${GATEWAY_URL}/api/v1/drugs/conditions`);
    const data = await res.json();

    if (!res.ok) {
      conditionCountEl.textContent = `Could not load conditions: ${data.detail || data.error || "unknown error"}`;
      recommendButton.disabled = true;
      return;
    }

    allConditions = data.conditions || [];
    conditionOptionsEl.innerHTML = allConditions.map((c) => `<option value="${escapeHtml(c)}"></option>`).join("");
    conditionCountEl.textContent = `${allConditions.length} conditions available — start typing to see suggestions`;
  } catch (err) {
    conditionCountEl.textContent = `Could not reach the API gateway at ${GATEWAY_URL}.`;
    recommendButton.disabled = true;
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const condition = conditionInput.value.trim();
  if (!condition) {
    showError("Enter a condition.");
    return;
  }

  const payload = {
    condition,
    age: ageInput.value ? Number(ageInput.value) : null,
    current_medications: parseCommaList(medsInput.value),
    allergies: parseCommaList(allergiesInput.value),
  };

  showLoading();

  try {
    const res = await fetch(`${GATEWAY_URL}/api/v1/drugs/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (res.status === 503) {
      showError(`The drug recommendation data isn't available right now: ${data.detail || ""}`);
      return;
    }
    if (res.status === 422) {
      showError(data.detail || "There's a problem with the request.");
      return;
    }
    if (!res.ok) {
      showError(data.detail || data.error || "Request failed");
      return;
    }

    showResult(data);
  } catch (err) {
    showError("Could not reach the API gateway. Is it running at " + GATEWAY_URL + "?");
  }
});

function showLoading() {
  resultEl.classList.remove("hidden", "error");
  resultEl.innerHTML = "<p>Looking up recommendations…</p>";
}

function showError(message) {
  resultEl.classList.remove("hidden");
  resultEl.classList.add("error");
  resultEl.innerHTML = `<h2>Something went wrong</h2><p>${escapeHtml(message)}</p>`;
}

function showResult(data) {
  resultEl.classList.remove("hidden", "error");

  // No match — point the user toward real vocabulary rather than leaving
  // them guessing why nothing came back.
  if (!data.condition_matched) {
    const suggestions = allConditions.slice(0, 8).map(escapeHtml).join(", ");
    resultEl.innerHTML = `
      <h2>No data for "${escapeHtml(data.condition)}"</h2>
      <p>This condition isn't in the dataset. A few examples that are: ${suggestions}${allConditions.length > 8 ? ", …" : ""}</p>
    `;
    return;
  }

  const rulesWarning = !data.contraindication_rules_loaded
    ? `<p class="rules-warning">⚠ No contraindication rules are loaded in this session — every "not flagged" result below is <strong>unchecked</strong>, not confirmed safe.</p>`
    : "";

  const sorted = [...data.recommendations].sort((a, b) => b.score - a.score);

  const rowsHtml = sorted
    .map((rec) => {
      const pct = ((rec.score / 10) * 100).toFixed(1);
      const positiveClass = rec.contraindicated ? "finding-positive" : "";
      const reasons = rec.contraindication_reasons.length
        ? `<div class="contraindication-reasons">${rec.contraindication_reasons.map(escapeHtml).join("; ")}</div>`
        : "";
      return `
        <li class="finding-row ${positiveClass}">
          <span class="finding-name">${escapeHtml(rec.drug)}</span>
          <span class="finding-bar-track">
            <span class="finding-bar-fill" style="width: ${pct}%"></span>
          </span>
          <span class="finding-pct">${rec.score.toFixed(1)}/10</span>
        </li>
        ${reasons}
      `;
    })
    .join("");

  resultEl.innerHTML = `
    <h2>${escapeHtml(data.condition_matched)}</h2>
    ${rulesWarning}
    <ul class="findings-list">${rowsHtml}</ul>
    <p><small>${escapeHtml(data.disclaimer)}</small></p>
    <p><small>Model version: ${escapeHtml(data.model_version)}</small></p>
  `;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

loadConditions();