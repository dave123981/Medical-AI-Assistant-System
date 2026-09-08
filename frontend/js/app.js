const GATEWAY_URL = window.GATEWAY_URL || "http://localhost:8080";

const form = document.getElementById("diagnosis-form");
const resultEl = document.getElementById("result");
const checklistEl = document.getElementById("symptom-checklist");
const filterInput = document.getElementById("symptom-filter");
const countEl = document.getElementById("symptom-count");
const selectedEl = document.getElementById("selected-symptoms");
const selectedCountEl = document.getElementById("selected-count");
const submitButton = form.querySelector("button[type='submit']");

let allSymptoms = [];
const selectedSymptoms = new Set();

function displayLabel(symptom) {
  // "skin_rash" -> "Skin rash" — purely cosmetic; the checkbox's
  // underlying value stays the exact vocab string sent to the backend.
  const spaced = symptom.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function renderChecklist(symptoms) {
  if (symptoms.length === 0) {
    checklistEl.innerHTML = '<p class="empty-text">No symptoms available. Is the model loaded?</p>';
    return;
  }

  checklistEl.innerHTML = symptoms
    .map(
      (symptom) => `
      <label data-symptom="${symptom}">
        <input type="checkbox" name="symptom" value="${symptom}" />
        ${displayLabel(symptom)}
      </label>
    `
    )
    .join("");
}

function renderSelected() {
  selectedCountEl.textContent = selectedSymptoms.size;

  if (selectedSymptoms.size === 0) {
    selectedEl.innerHTML = '<p class="empty-text">None selected yet.</p>';
    return;
  }

  selectedEl.innerHTML = Array.from(selectedSymptoms)
    .map(
      (symptom) => `
      <span class="symptom-chip" data-symptom="${symptom}">
        ${displayLabel(symptom)}
        <button type="button" class="chip-remove" data-symptom="${symptom}" aria-label="Remove ${escapeHtml(
          displayLabel(symptom)
        )}">×</button>
      </span>
    `
    )
    .join("");
}

// Event delegation: one listener handles every checkbox, including ones
// added/re-filtered after initial render.
checklistEl.addEventListener("change", (e) => {
  if (e.target.name !== "symptom") return;

  if (e.target.checked) {
    selectedSymptoms.add(e.target.value);
  } else {
    selectedSymptoms.delete(e.target.value);
  }
  renderSelected();
});

// Clicking the × on a chip unchecks the matching checkbox and updates
// the selected set — the two views stay in sync either direction.
selectedEl.addEventListener("click", (e) => {
  if (!e.target.classList.contains("chip-remove")) return;

  const symptom = e.target.dataset.symptom;
  selectedSymptoms.delete(symptom);

  const checkbox = checklistEl.querySelector(`input[value="${cssEscape(symptom)}"]`);
  if (checkbox) checkbox.checked = false;

  renderSelected();
});

function applyFilter() {
  const query = filterInput.value.trim().toLowerCase();
  const labels = checklistEl.querySelectorAll("label[data-symptom]");
  let visibleCount = 0;

  labels.forEach((label) => {
    const matches = label.dataset.symptom.replace(/_/g, " ").includes(query);
    label.classList.toggle("hidden", !matches);
    if (matches) visibleCount++;
  });

  countEl.textContent = query
    ? `Showing ${visibleCount} of ${allSymptoms.length} symptoms`
    : `${allSymptoms.length} symptoms available`;
}

async function loadSymptoms() {
  try {
    const res = await fetch(`${GATEWAY_URL}/api/v1/diagnosis/symptoms`);
    const data = await res.json();

    if (!res.ok) {
      checklistEl.innerHTML = `<p class="empty-text">Could not load symptoms: ${escapeHtml(
        data.detail || data.error || "unknown error"
      )}</p>`;
      submitButton.disabled = true;
      return;
    }

    allSymptoms = data.symptoms || [];
    renderChecklist(allSymptoms);
    countEl.textContent = `${allSymptoms.length} symptoms available`;
  } catch (err) {
    checklistEl.innerHTML = `<p class="empty-text">Could not reach the API gateway at ${GATEWAY_URL}.</p>`;
    submitButton.disabled = true;
  }
}

filterInput.addEventListener("input", applyFilter);

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const age = Number(document.getElementById("age").value);
  const gender = document.getElementById("gender").value;
  const symptoms = Array.from(selectedSymptoms);

  if (symptoms.length === 0) {
    showError("Select at least one symptom.");
    return;
  }

  showLoading();

  try {
    const res = await fetch(`${GATEWAY_URL}/api/v1/diagnosis/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        age,
        gender,
        symptoms,
        vital_signs: null,
        lab_values: null,
        medical_history: [],
      }),
    });

    const data = await res.json();

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
  resultEl.innerHTML = "<p>Predicting…</p>";
}

function showError(message) {
  resultEl.classList.remove("hidden");
  resultEl.classList.add("error");
  resultEl.innerHTML = `<h2>Something went wrong</h2><p>${escapeHtml(message)}</p>`;
}

function showResult(data) {
  resultEl.classList.remove("hidden", "error");

  const candidates = (data.top_candidates || [])
    .map(
      (c) =>
        `<li><span>${escapeHtml(c.disease)}</span><span>${(c.probability * 100).toFixed(1)}%</span></li>`
    )
    .join("");

  const precautions = (data.precautions || []).map((p) => `<li>${escapeHtml(p)}</li>`).join("");

  resultEl.innerHTML = `
    <h2>${escapeHtml(data.predicted_disease)} <small>(${(data.confidence * 100).toFixed(1)}% confidence)</small></h2>
    ${data.description ? `<p>${escapeHtml(data.description)}</p>` : ""}
    ${candidates ? `<h3>Other possibilities</h3><ul class="candidate-list">${candidates}</ul>` : ""}
    ${precautions ? `<h3>Suggested precautions</h3><ul>${precautions}</ul>` : ""}
    <p><small>Model version: ${escapeHtml(data.model_version)}</small></p>
  `;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function cssEscape(value) {
  // Minimal escape for use inside a querySelector attribute value —
  // symptom names are plain snake_case so this only needs to guard
  // against characters that would break the selector syntax itself.
  return value.replace(/(["\\])/g, "\\$1");
}

loadSymptoms();

