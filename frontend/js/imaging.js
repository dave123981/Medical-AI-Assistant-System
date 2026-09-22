const GATEWAY_URL = window.GATEWAY_URL || "http://localhost:8080";

const form = document.getElementById("imaging-form");
const resultEl = document.getElementById("result");
const imageTypeSelect = document.getElementById("image-type");
const conditionsNoteEl = document.getElementById("conditions-note");
const fileInput = document.getElementById("image-file");
const previewWrap = document.getElementById("image-preview-wrap");
const previewImg = document.getElementById("image-preview");
const thresholdInput = document.getElementById("threshold");
const thresholdValueEl = document.getElementById("threshold-value");
const analyzeButton = document.getElementById("analyze-button");

function displayLabel(name) {
  return name.replace(/_/g, " ");
}

// Loads and displays which conditions the active model for the selected
// image type actually predicts. A 501 here (image type not built yet)
// disables submission entirely rather than letting the user hit a dead
// end after uploading an image.
async function loadConditions() {
  const imageType = imageTypeSelect.value;
  conditionsNoteEl.textContent = "Loading…";
  analyzeButton.disabled = false;

  try {
    const res = await fetch(
      `${GATEWAY_URL}/api/v1/imaging/conditions?image_type=${encodeURIComponent(imageType)}`
    );
    const data = await res.json();

    if (res.status === 501) {
      conditionsNoteEl.textContent = `${displayLabel(imageType)} isn't built yet — analysis is disabled for this image type.`;
      analyzeButton.disabled = true;
      return;
    }

    if (!res.ok) {
      conditionsNoteEl.textContent = `Could not load conditions: ${data.detail || data.error || "unknown error"}`;
      analyzeButton.disabled = true;
      return;
    }

    const conditions = (data.conditions || []).map(displayLabel).join(", ");
    conditionsNoteEl.textContent = `This model checks for: ${conditions}`;
  } catch (err) {
    conditionsNoteEl.textContent = `Could not reach the API gateway at ${GATEWAY_URL}.`;
    analyzeButton.disabled = true;
  }
}

imageTypeSelect.addEventListener("change", loadConditions);

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) {
    previewWrap.classList.add("hidden");
    previewImg.src = "";
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewWrap.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
});

thresholdInput.addEventListener("input", () => {
  thresholdValueEl.textContent = Number(thresholdInput.value).toFixed(2);
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    showError("Choose an image to upload.");
    return;
  }

  const formData = new FormData();
  formData.append("image", file);
  formData.append("image_type", imageTypeSelect.value);
  formData.append("threshold", thresholdInput.value);

  showLoading();

  try {
    const res = await fetch(`${GATEWAY_URL}/api/v1/imaging/analyze`, {
      method: "POST",
      body: formData, // no Content-Type header — browser sets the multipart boundary itself
    });

    const data = await res.json();

    if (res.status === 501) {
      showError(`This image type isn't supported yet: ${data.detail || ""}`);
      return;
    }
    if (res.status === 422) {
      showError(`There's a problem with the upload: ${data.detail || ""}`);
      return;
    }
    if (res.status === 503) {
      showError(`The imaging model isn't available right now: ${data.detail || ""}`);
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
  resultEl.innerHTML = "<p>Analyzing…</p>";
}

function showError(message) {
  resultEl.classList.remove("hidden");
  resultEl.classList.add("error");
  resultEl.innerHTML = `<h2>Something went wrong</h2><p>${escapeHtml(message)}</p>`;
}

function showResult(data) {
  resultEl.classList.remove("hidden", "error");

  // Sort by probability descending so the most notable findings are visible
  // first without the user having to scan all 14.
  const sortedFindings = [...data.findings].sort((a, b) => b.probability - a.probability);

  const findingsHtml = sortedFindings
    .map((f) => {
      const pct = (f.probability * 100).toFixed(1);
      const positiveClass = f.positive ? "finding-positive" : "";
      return `
        <li class="finding-row ${positiveClass}">
          <span class="finding-name">${escapeHtml(displayLabel(f.condition))}</span>
          <span class="finding-bar-track">
            <span class="finding-bar-fill" style="width: ${pct}%"></span>
          </span>
          <span class="finding-pct">${pct}%</span>
        </li>
      `;
    })
    .join("");

  const positiveSummary = data.positive_findings.length
    ? data.positive_findings.map(displayLabel).join(", ")
    : "None above threshold (No Finding)";

  resultEl.innerHTML = `
    <h2>${escapeHtml(positiveSummary)}</h2>
    <p><small>Threshold: ${data.threshold} — model version: ${escapeHtml(data.model_version)}</small></p>
    <ul class="findings-list">${findingsHtml}</ul>
  `;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

loadConditions();
