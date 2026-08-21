// Points at the Go API Gateway, not the Python service directly.
// Override via a query param or hardcode for your deployment, e.g.
// http://localhost:8080 for local dev.
const GATEWAY_URL = window.GATEWAY_URL || "http://localhost:8080";

const form = document.getElementById("diagnosis-form");
const resultEl = document.getElementById("result");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const age = Number(document.getElementById("age").value);
  const gender = document.getElementById("gender").value;
  const symptomsRaw = document.getElementById("symptoms").value;
  const symptoms = symptomsRaw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

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
