package clients

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"time"
)

// ChatbotClient talks to the Python medical-chatbot service. Same
// status-code-passthrough pattern as DrugClient/ImagingClient.
type ChatbotClient struct {
	BaseURL string
	http    *http.Client
}

func NewChatbotClient(baseURL string) *ChatbotClient {
	return &ChatbotClient{
		BaseURL: baseURL,
		http:    &http.Client{Timeout: 10 * time.Second},
	}
}

func (c *ChatbotClient) Ask(ctx context.Context, body []byte) (int, []byte, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/ask", bytes.NewReader(body))
	if err != nil {
		return 0, nil, fmt.Errorf("building request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return 0, nil, fmt.Errorf("calling chatbot service: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, nil, fmt.Errorf("reading response: %w", err)
	}

	return resp.StatusCode, respBody, nil
}