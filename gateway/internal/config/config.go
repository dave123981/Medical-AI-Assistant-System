package config

import "os"

// Config holds all environment-driven settings for the gateway.
// Every downstream microservice gets its own base URL so the gateway
// never hardcodes service locations — this is what lets you swap a
// stubbed service for a real one without touching handler code.
type Config struct {
	Port                string
	DiagnosisServiceURL  string
	ImagingServiceURL    string
	DrugServiceURL       string
	ChatbotServiceURL    string
	AllowedOrigins       string
}

func Load() *Config {
	return &Config{
		Port:                getEnv("PORT", "8080"),
		DiagnosisServiceURL: getEnv("DIAGNOSIS_SERVICE_URL", "http://localhost:8000"),
		ImagingServiceURL:   getEnv("IMAGING_SERVICE_URL", ""),  // not implemented yet
		DrugServiceURL:      getEnv("DRUG_SERVICE_URL", ""),     // not implemented yet
		ChatbotServiceURL:   getEnv("CHATBOT_SERVICE_URL", ""),  // not implemented yet
		AllowedOrigins:      getEnv("ALLOWED_ORIGINS", "*"),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
