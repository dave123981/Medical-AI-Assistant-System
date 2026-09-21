package handlers

import "net/http"

func Health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func StubDrugRecommend(w http.ResponseWriter, r *http.Request) {
	writeNotImplemented(w, "drug_service", "Drug Recommendation Assistant (Service 3) is not implemented yet.")
}

func StubChatbotAsk(w http.ResponseWriter, r *http.Request) {
	writeNotImplemented(w, "chatbot_service", "Medical Chatbot (Service 4) is not implemented yet.")
}

func writeNotImplemented(w http.ResponseWriter, service, detail string) {
	writeJSON(w, http.StatusNotImplemented, map[string]string{
		"error":  service + "_not_implemented",
		"detail": detail,
	})
}
