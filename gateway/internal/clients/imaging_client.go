package clients

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"time"
)

// ImagingClient forwards multipart image uploads to the Python

type ImagingClient struct {
	BaseURL string
	http    *http.Client
}

func NewImagingClient(baseURL string) *ImagingClient {
	return &ImagingClient{
		BaseURL: baseURL,
		// Longer than the diagnosis client's timeout: image upload + a CNN
		// forward pass takes more time than a JSON symptom lookup.
		http: &http.Client{Timeout: 30 * time.Second},
	}
}


func (c *ImagingClient) Analyze(ctx context.Context, file io.Reader, filename, imageType string, threshold float64) (int, []byte, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("image", filename)
	if err != nil {
		return 0, nil, fmt.Errorf("creating multipart file field: %w", err)
	}
	if _, err := io.Copy(part, file); err != nil {
		return 0, nil, fmt.Errorf("copying image content: %w", err)
	}
	if err := writer.WriteField("image_type", imageType); err != nil {
		return 0, nil, fmt.Errorf("writing image_type field: %w", err)
	}
	if err := writer.WriteField("threshold", fmt.Sprintf("%v", threshold)); err != nil {
		return 0, nil, fmt.Errorf("writing threshold field: %w", err)
	}
	if err := writer.Close(); err != nil {
		return 0, nil, fmt.Errorf("closing multipart writer: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/analyze", body)
	if err != nil {
		return 0, nil, fmt.Errorf("building request: %w", err)
	}
	httpReq.Header.Set("Content-Type", writer.FormDataContentType())

	return c.doAndRead(httpReq)
}

// GetConditions proxies GET /conditions?image_type=... for a given image type.
func (c *ImagingClient) GetConditions(ctx context.Context, imageType string) (int, []byte, error) {
	httpReq, err := http.NewRequestWithContext(
		ctx, http.MethodGet,
		c.BaseURL+"/conditions?image_type="+url.QueryEscape(imageType),
		nil,
	)
	if err != nil {
		return 0, nil, fmt.Errorf("building request: %w", err)
	}

	return c.doAndRead(httpReq)
}

func (c *ImagingClient) doAndRead(httpReq *http.Request) (int, []byte, error) {
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return 0, nil, fmt.Errorf("calling imaging service: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, nil, fmt.Errorf("reading response: %w", err)
	}

	return resp.StatusCode, respBody, nil
}
