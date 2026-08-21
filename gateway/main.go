package main

import (
	"log"
	"net/http"

	"github.com/ashthecoder05/medical-ai-gateway/internal/config"
	"github.com/ashthecoder05/medical-ai-gateway/internal/router"
)

func main() {
	cfg := config.Load()
	handler := router.New(cfg)

	log.Printf("API Gateway listening on :%s", cfg.Port)
	log.Printf("Diagnosis service -> %s", cfg.DiagnosisServiceURL)
	if err := http.ListenAndServe(":"+cfg.Port, handler); err != nil {
		log.Fatalf("server failed: %v", err)
	}
}
