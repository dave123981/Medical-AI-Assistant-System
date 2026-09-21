package config

import "os"

type Config struct {
	Port                string
	DiagnosisServiceURL string
	ImagingServiceURL   string
	AllowedOrigins      string
}

func Load() *Config {
	return &Config{
		Port:                getEnv("PORT", "8080"),
		DiagnosisServiceURL: getEnv("DIAGNOSIS_SERVICE_URL", "http://localhost:8000"),
		ImagingServiceURL:   getEnv("IMAGING_SERVICE_URL", "http://localhost:8001"),
		AllowedOrigins:      getEnv("ALLOWED_ORIGINS", "*"),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
