package router

import (
	"net/http"

	"github.com/ashthecoder05/medical-ai-gateway/internal/clients"
	"github.com/ashthecoder05/medical-ai-gateway/internal/config"
	"github.com/ashthecoder05/medical-ai-gateway/internal/handlers"
	"github.com/ashthecoder05/medical-ai-gateway/internal/middleware"
)

func New(cfg *config.Config) http.Handler {
	mux := http.NewServeMux()

	diagnosisClient := clients.NewDiagnosisClient(cfg.DiagnosisServiceURL)
	diagnosisHandler := handlers.NewDiagnosisHandler(diagnosisClient)

	mux.HandleFunc("GET /api/v1/health", handlers.Health)

	// Service 1 — fully wired
	mux.HandleFunc("GET /api/v1/diagnosis/symptoms", diagnosisHandler.GetSymptoms)
	mux.HandleFunc("POST /api/v1/diagnosis/predict", diagnosisHandler.Predict)

	// Services 2-4 — stubbed on purpose
	mux.HandleFunc("POST /api/v1/imaging/analyze", handlers.StubImagingAnalyze)
	mux.HandleFunc("POST /api/v1/drugs/recommend", handlers.StubDrugRecommend)
	mux.HandleFunc("POST /api/v1/chatbot/ask", handlers.StubChatbotAsk)

	return middleware.Chain(mux,
		middleware.Logging,
		middleware.CORS(cfg.AllowedOrigins),
	)
}
