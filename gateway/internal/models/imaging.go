package models

type ConditionProbability struct {
	Condition   string  `json:"condition"`
	Probability float64 `json:"probability"`
	Positive    bool    `json:"positive"`
}

type ImageAnalysisResponse struct {
	ImageType        string                 `json:"image_type"`
	Findings         []ConditionProbability `json:"findings"`
	PositiveFindings []string               `json:"positive_findings"`
	Threshold        float64                `json:"threshold"`
	HeatmapBase64    *string                `json:"heatmap_base64,omitempty"`
	ModelVersion     string                 `json:"model_version"`
}

type ConditionsResponse struct {
	ImageType  string   `json:"image_type"`
	Conditions []string `json:"conditions"`
	Count      int      `json:"count"`
}
