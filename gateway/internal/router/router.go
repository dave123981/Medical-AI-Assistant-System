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

	imagingClient := clients.NewImagingClient(cfg.ImagingServiceURL)
	imagingHandler := handlers.NewImagingHandler(imagingClient)

	drugClient := clients.NewDrugClient(cfg.DrugServiceURL)
	drugHandler := handlers.NewDrugHandler(drugClient)

	chatbotClient := clients.NewChatbotClient(cfg.ChatbotServiceURL)
	chatbotHandler := handlers.NewChatbotHandler(chatbotClient)

	mux.HandleFunc("GET /api/v1/health", handlers.Health)

	// Service 1 — fully wired
	mux.HandleFunc("GET /api/v1/diagnosis/symptoms", diagnosisHandler.GetSymptoms)
	mux.HandleFunc("POST /api/v1/diagnosis/predict", diagnosisHandler.Predict)

	// Service 2 — fully wired
	mux.HandleFunc("GET /api/v1/imaging/conditions", imagingHandler.GetConditions)
	mux.HandleFunc("POST /api/v1/imaging/analyze", imagingHandler.Analyze)

	// Service 3 — fully wired
	mux.HandleFunc("GET /api/v1/drugs/conditions", drugHandler.GetConditions)
	mux.HandleFunc("POST /api/v1/drugs/recommend", drugHandler.Recommend)

	// Service 4 — fully wired
	mux.HandleFunc("POST /api/v1/chatbot/ask", chatbotHandler.Ask)

	return middleware.Chain(mux,
		middleware.Logging,
		middleware.CORS(cfg.AllowedOrigins),
	)
}