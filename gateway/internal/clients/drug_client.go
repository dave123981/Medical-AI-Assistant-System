package clients

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"time"
)

// DrugClient talks to the Python drug-recommendation service. Uses the
// same status-code-passthrough pattern as ImagingClient (not the older
// DiagnosisClient style) — the Python service distinguishes 503 (missing
// data files) from 200 with an empty recommendations list (condition not
// found), and both matter to the caller.
type DrugClient struct {
	BaseURL string
	http    *http.Client
}

func NewDrugClient(baseURL string) *DrugClient {
	return &DrugClient{
		BaseURL: baseURL,
		http:    &http.Client{Timeout: 10 * time.Second},
	}
}

func (c *DrugClient) Recommend(ctx context.Context, body []byte) (int, []byte, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/recommend", bytes.NewReader(body))
	if err != nil {
		return 0, nil, fmt.Errorf("building request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	return c.doAndRead(httpReq)
}

func (c *DrugClient) GetConditions(ctx context.Context) (int, []byte, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/conditions", nil)
	if err != nil {
		return 0, nil, fmt.Errorf("building request: %w", err)
	}

	return c.doAndRead(httpReq)
}

func (c *DrugClient) doAndRead(httpReq *http.Request) (int, []byte, error) {
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return 0, nil, fmt.Errorf("calling drug recommendation service: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, nil, fmt.Errorf("reading response: %w", err)
	}

	return resp.StatusCode, respBody, nil
}