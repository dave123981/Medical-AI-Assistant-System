package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/ashthecoder05/medical-ai-gateway/internal/clients"
	"github.com/ashthecoder05/medical-ai-gateway/internal/models"
)

type DrugHandler struct {
	Client *clients.DrugClient
}

func NewDrugHandler(client *clients.DrugClient) *DrugHandler {
	return &DrugHandler{Client: client}
}

// GetConditions handles GET /api/v1/drugs/conditions
func (h *DrugHandler) GetConditions(w http.ResponseWriter, r *http.Request) {
	status, body, err := h.Client.GetConditions(r.Context())
	if err != nil {
		writeError(w, http.StatusBadGateway, "drug_service_error", err.Error())
		return
	}
	relayUpstreamJSON(w, status, body)
}

// Recommend handles POST /api/v1/drugs/recommend
func (h *DrugHandler) Recommend(w http.ResponseWriter, r *http.Request) {
	var req models.RecommendationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_json", err.Error())
		return
	}

	if req.Condition == "" {
		writeError(w, http.StatusUnprocessableEntity, "validation_error", "condition is required")
		return
	}

	body, err := json.Marshal(req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "encoding_error", err.Error())
		return
	}

	status, respBody, err := h.Client.Recommend(r.Context(), body)
	if err != nil {
		writeError(w, http.StatusBadGateway, "drug_service_error", err.Error())
		return
	}
	relayUpstreamJSON(w, status, respBody)
}