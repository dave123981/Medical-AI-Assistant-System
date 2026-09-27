package models

type AskRequest struct {
	Question          string  `json:"question" validate:"required"`
	ContextDisease    *string `json:"context_disease,omitempty"`
	ConversationID    *string `json:"conversation_id,omitempty"`
}

type AskResponse struct {
	Answer           string   `json:"answer"`
	MatchedQuestion  string   `json:"matched_question"`
	SimilarityScore  float64  `json:"similarity_score"`
	Confident        bool     `json:"confident"`
	Sources          []string `json:"sources"`
	Disclaimer       string   `json:"disclaimer"`
	ModelVersion     string   `json:"model_version"`
}