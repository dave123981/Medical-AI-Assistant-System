package clients

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/ashthecoder05/medical-ai-gateway/internal/models"
)

type DiagnosisClient struct {
	BaseURL string
	http    *http.Client
}

func NewDiagnosisClient(baseURL string) *DiagnosisClient {
	return &DiagnosisClient{
		BaseURL: baseURL,
		http:    &http.Client{Timeout: 10 * time.Second},
	}
}

func (c *DiagnosisClient) Predict(ctx context.Context, req models.DiagnosisRequest) (*models.DiagnosisResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("encoding request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/predict", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("building request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("calling diagnosis service: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("diagnosis service returned %d: %s", resp.StatusCode, string(respBody))
	}

	var out models.DiagnosisResponse
	if err := json.Unmarshal(respBody, &out); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}
	return &out, nil
}

// GetSymptoms fetches the ordered symptom vocabulary from the diagnosis
// service, so the frontend can render a checklist instead of a free-text
// field. Uses a longer timeout than Predict since this is called once on
// page load, not on every keystroke, and the payload is larger.
func (c *DiagnosisClient) GetSymptoms(ctx context.Context) (*models.SymptomsResponse, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/symptoms", nil)
	if err != nil {
		return nil, fmt.Errorf("building request: %w", err)
	}

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("calling diagnosis service: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("diagnosis service returned %d: %s", resp.StatusCode, string(respBody))
	}

	var out models.SymptomsResponse
	if err := json.Unmarshal(respBody, &out); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}
	return &out, nil
}
