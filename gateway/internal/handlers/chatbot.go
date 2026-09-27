package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/ashthecoder05/medical-ai-gateway/internal/clients"
	"github.com/ashthecoder05/medical-ai-gateway/internal/models"
)

type ChatbotHandler struct {
	Client *clients.ChatbotClient
}

func NewChatbotHandler(client *clients.ChatbotClient) *ChatbotHandler {
	return &ChatbotHandler{Client: client}
}

// Ask handles POST /api/v1/chatbot/ask
func (h *ChatbotHandler) Ask(w http.ResponseWriter, r *http.Request) {
	var req models.AskRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_json", err.Error())
		return
	}

	if req.Question == "" {
		writeError(w, http.StatusUnprocessableEntity, "validation_error", "question is required")
		return
	}

	body, err := json.Marshal(req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "encoding_error", err.Error())
		return
	}

	status, respBody, err := h.Client.Ask(r.Context(), body)
	if err != nil {
		writeError(w, http.StatusBadGateway, "chatbot_service_error", err.Error())
		return
	}
	relayUpstreamJSON(w, status, respBody)
}