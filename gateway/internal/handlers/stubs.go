package handlers

import "net/http"

// Health handles GET /api/v1/health
func Health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// The handlers below are intentional placeholders. They exist now so the
// route table, request logging, and CORS behavior are already correct
// when each real service gets built — you're only ever swapping the
// handler body, never the routing layer.

// StubImagingAnalyze — POST /api/v1/imaging/analyze
// Planned input:  multipart image (chest X-ray / skin lesion / retinal) + image_type
// Planned output: { finding: string, confidence: float, heatmap_url?: string, model_version: string }
func StubImagingAnalyze(w http.ResponseWriter, r *http.Request) {
	writeNotImplemented(w, "imaging_service", "Medical Image Analysis (Service 2) is not implemented yet.")
}

// StubDrugRecommend — POST /api/v1/drugs/recommend
// Planned input:  { diagnosis: string, age: int, current_medications: []string, allergies: []string }
// Planned output: { recommendations: [{drug, score, contraindications}], model_version: string }
func StubDrugRecommend(w http.ResponseWriter, r *http.Request) {
	writeNotImplemented(w, "drug_service", "Drug Recommendation Assistant (Service 3) is not implemented yet.")
}

// StubChatbotAsk — POST /api/v1/chatbot/ask
// Planned input:  { question: string, context_disease?: string, conversation_id?: string }
// Planned output: { answer: string, sources: []string, model_version: string }
func StubChatbotAsk(w http.ResponseWriter, r *http.Request) {
	writeNotImplemented(w, "chatbot_service", "Medical Chatbot (Service 4) is not implemented yet.")
}

func writeNotImplemented(w http.ResponseWriter, service, detail string) {
	writeJSON(w, http.StatusNotImplemented, map[string]string{
		"error":   service + "_not_implemented",
		"detail":  detail,
	})
}
