package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/ashthecoder05/medical-ai-gateway/internal/clients"
	"github.com/ashthecoder05/medical-ai-gateway/internal/models"
)

type DiagnosisHandler struct {
	Client *clients.DiagnosisClient
}

func NewDiagnosisHandler(client *clients.DiagnosisClient) *DiagnosisHandler {
	return &DiagnosisHandler{Client: client}
}

// Predict handles POST /api/v1/diagnosis/predict
func (h *DiagnosisHandler) Predict(w http.ResponseWriter, r *http.Request) {
	var req models.DiagnosisRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_json", err.Error())
		return
	}

	if problem := validateDiagnosisRequest(req); problem != "" {
		writeError(w, http.StatusUnprocessableEntity, "validation_error", problem)
		return
	}

	resp, err := h.Client.Predict(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusBadGateway, "diagnosis_service_error", err.Error())
		return
	}

	writeJSON(w, http.StatusOK, resp)
}

func validateDiagnosisRequest(req models.DiagnosisRequest) string {
	if req.Age < 0 || req.Age > 120 {
		return "age must be between 0 and 120"
	}
	if req.Gender != "male" && req.Gender != "female" && req.Gender != "other" {
		return "gender must be one of: male, female, other"
	}
	if len(req.Symptoms) == 0 {
		return "at least one symptom is required"
	}
	return ""
}

func writeJSON(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, errCode, detail string) {
	writeJSON(w, status, models.ErrorResponse{Error: errCode, Detail: detail})
}
