package models

type RecommendationRequest struct {
	Condition          string   `json:"condition" validate:"required"`
	Age                *int     `json:"age,omitempty"`
	CurrentMedications []string `json:"current_medications,omitempty"`
	Allergies          []string `json:"allergies,omitempty"`
}

type DrugRecommendation struct {
	Drug                    string   `json:"drug"`
	Score                   float64  `json:"score"`
	ReviewCount             int      `json:"review_count"`
	Contraindicated         bool     `json:"contraindicated"`
	ContraindicationReasons []string `json:"contraindication_reasons"`
}

type RecommendationResponse struct {
	Condition                   string                `json:"condition"`
	ConditionMatched            *string               `json:"condition_matched"`
	Recommendations             []DrugRecommendation  `json:"recommendations"`
	ContraindicationRulesLoaded bool                  `json:"contraindication_rules_loaded"`
	Disclaimer                  string                `json:"disclaimer"`
	ModelVersion                string                `json:"model_version"`
}

type DrugConditionsResponse struct {
	Conditions []string `json:"conditions"`
	Count      int      `json:"count"`
}