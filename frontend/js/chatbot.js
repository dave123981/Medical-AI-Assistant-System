const GATEWAY_URL = window.GATEWAY_URL || "http://localhost:8080";

const chatForm = document.getElementById("chat-form");
const chatLog = document.getElementById("chat-log");
const questionInput = document.getElementById("question-input");
const contextDiseaseInput = document.getElementById("context-disease");
const askButton = document.getElementById("ask-button");

// Generated once per page load, accepted-but-unused by the v1 service —
// included now for forward compatibility with future conversation memory.
const conversationId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());

let hasStarted = false;

function clearEmptyState() {
  if (!hasStarted) {
    chatLog.innerHTML = "";
    hasStarted = true;
  }
}

function appendUserMessage(question) {
  clearEmptyState();
  const el = document.createElement("div");
  el.className = "chat-message chat-message-user";
  el.innerHTML = `<div class="chat-bubble chat-bubble-user">${escapeHtml(question)}</div>`;
  chatLog.appendChild(el);
  scrollToBottom();
}

function appendLoadingMessage() {
  const el = document.createElement("div");
  el.className = "chat-message chat-message-assistant";
  el.id = "loading-message";
  el.innerHTML = `<div class="chat-bubble chat-bubble-assistant">Thinking…</div>`;
  chatLog.appendChild(el);
  scrollToBottom();
}

function removeLoadingMessage() {
  const el = document.getElementById("loading-message");
  if (el) el.remove();
}

function appendAssistantMessage(data) {
  const el = document.createElement("div");
  el.className = "chat-message chat-message-assistant";

  const confidenceNote = !data.confident
    ? `<div class="chat-confidence-warning">⚠ Low-confidence match — this may not directly answer your question.</div>`
    : "";

  el.innerHTML = `
    <div class="chat-bubble chat-bubble-assistant">
      ${escapeHtml(data.answer)}
      ${confidenceNote}
      <div class="chat-meta">
        Matched: "${escapeHtml(data.matched_question)}" (similarity: ${data.similarity_score.toFixed(2)})
      </div>
      <div class="chat-meta">${escapeHtml(data.disclaimer)}</div>
      <div class="chat-meta">Sources: ${data.sources.map(escapeHtml).join(", ")} — model: ${escapeHtml(data.model_version)}</div>
    </div>
  `;
  chatLog.appendChild(el);
  scrollToBottom();
}

function appendErrorMessage(message) {
  const el = document.createElement("div");
  el.className = "chat-message chat-message-assistant";
  el.innerHTML = `<div class="chat-bubble chat-bubble-error">${escapeHtml(message)}</div>`;
  chatLog.appendChild(el);
  scrollToBottom();
}

function scrollToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const question = questionInput.value.trim();
  if (!question) return;

  appendUserMessage(question);
  questionInput.value = "";
  askButton.disabled = true;
  appendLoadingMessage();

  const payload = {
    question,
    context_disease: contextDiseaseInput.value.trim() || null,
    conversation_id: conversationId,
  };

  try {
    const res = await fetch(`${GATEWAY_URL}/api/v1/chatbot/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    removeLoadingMessage();

    if (res.status === 503) {
      appendErrorMessage(`The chatbot data isn't available right now: ${data.detail || ""}`);
      return;
    }
    if (res.status === 422) {
      appendErrorMessage(data.detail || "There's a problem with the question.");
      return;
    }
    if (!res.ok) {
      appendErrorMessage(data.detail || data.error || "Request failed");
      return;
    }

    appendAssistantMessage(data);
  } catch (err) {
    removeLoadingMessage();
    appendErrorMessage("Could not reach the API gateway. Is it running at " + GATEWAY_URL + "?");
  } finally {
    askButton.disabled = false;
  }
});

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}