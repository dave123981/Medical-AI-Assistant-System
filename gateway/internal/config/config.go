package config

import "os"

type Config struct {
	Port                string
	DiagnosisServiceURL string
	ImagingServiceURL   string
	DrugServiceURL      string
	AllowedOrigins      string
}

func Load() *Config {
	return &Config{
		Port:                getEnv("PORT", "8080"),
		DiagnosisServiceURL: getEnv("DIAGNOSIS_SERVICE_URL", "http://localhost:8000"),
		ImagingServiceURL:   getEnv("IMAGING_SERVICE_URL", "http://localhost:8001"),
		DrugServiceURL:      getEnv("DRUG_SERVICE_URL", "http://localhost:8002"),
		AllowedOrigins:      getEnv("ALLOWED_ORIGINS", "*"),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}