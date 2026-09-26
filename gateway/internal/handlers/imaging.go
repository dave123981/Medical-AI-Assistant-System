package handlers

import (
	"net/http"
	"strconv"

	"github.com/ashthecoder05/medical-ai-gateway/internal/clients"
)

type ImagingHandler struct {
	Client *clients.ImagingClient
}

func NewImagingHandler(client *clients.ImagingClient) *ImagingHandler {
	return &ImagingHandler{Client: client}
}

// 20MB is generous for a single medical image (chest X-rays are often
// 1-3MB even at full resolution; this leaves headroom without allowing
// arbitrarily large uploads to tie up the gateway).
const maxUploadSize = 20 << 20

// GetConditions handles GET /api/v1/imaging/conditions
func (h *ImagingHandler) GetConditions(w http.ResponseWriter, r *http.Request) {
	imageType := r.URL.Query().Get("image_type")
	if imageType == "" {
		imageType = "chest_xray"
	}

	status, body, err := h.Client.GetConditions(r.Context(), imageType)
	if err != nil {
		writeError(w, http.StatusBadGateway, "imaging_service_error", err.Error())
		return
	}
	relayUpstreamJSON(w, status, body)
}

// Analyze handles POST /api/v1/imaging/analyze
func (h *ImagingHandler) Analyze(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(maxUploadSize); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_form", "Could not parse the upload: "+err.Error())
		return
	}

	file, fileHeader, err := r.FormFile("image")
	if err != nil {
		writeError(w, http.StatusBadRequest, "missing_image", "An 'image' file field is required.")
		return
	}
	defer file.Close()

	imageType := r.FormValue("image_type")
	if imageType == "" {
		imageType = "chest_xray"
	}

	// nil means "not provided" — let the Python service default to its
	// per-class tuned thresholds. Only build a pointer when the caller
	// actually sent a value, so we never silently inject a threshold the
	// frontend didn't ask for.
	var threshold *float64
	if v := r.FormValue("threshold"); v != "" {
		parsed, err := strconv.ParseFloat(v, 64)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid_threshold", "threshold must be a number.")
			return
		}
		threshold = &parsed
	}

	status, body, err := h.Client.Analyze(r.Context(), file, fileHeader.Filename, imageType, threshold)
	if err != nil {
		writeError(w, http.StatusBadGateway, "imaging_service_error", err.Error())
		return
	}
	relayUpstreamJSON(w, status, body)
}

// relayUpstreamJSON forwards the Python service's exact status code and
// JSON body to the client, preserving the 422/501/503 distinctions the
// imaging service was specifically designed to make — see ImagingClient's
// doc comment for why this differs from the diagnosis handler's approach.
func relayUpstreamJSON(w http.ResponseWriter, status int, body []byte) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}
