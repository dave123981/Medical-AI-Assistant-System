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

// DiagnosisClient talks to the Python FastAPI disease-diagnosis service.
// The gateway never runs the model itself — it only forwards the
// validated request and relays the response. This keeps Go/Python
// concerns cleanly separated: Go owns routing, validation, and the
// public contract; Python owns the model.
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
