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

	threshold := 0.5
	if v := r.FormValue("threshold"); v != "" {
		parsed, err := strconv.ParseFloat(v, 64)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid_threshold", "threshold must be a number.")
			return
		}
		threshold = parsed
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
