package models

// DiagnosisRequest is the canonical input contract for Service 1.
// It intentionally carries every field listed in the system spec
// (age, gender, symptoms, vitals, labs, history) even though the
// v1 Kaggle "Disease Symptom Prediction" dataset only contains
// Disease + Symptom columns. Only Symptoms is consumed by the v1
// model today; the rest are validated, stored, and passed through
// so that v2+ models (which may be trained on richer data) can
// start using them without breaking the API shape.
type DiagnosisRequest struct {
	Age            int                `json:"age" validate:"required,gte=0,lte=120"`
	Gender         string             `json:"gender" validate:"required,oneof=male female other"`
	Symptoms       []string           `json:"symptoms" validate:"required,min=1"`
	VitalSigns     *VitalSigns        `json:"vital_signs,omitempty"`
	LabValues      map[string]float64 `json:"lab_values,omitempty"`
	MedicalHistory []string           `json:"medical_history,omitempty"`
}

type VitalSigns struct {
	TemperatureC    *float64 `json:"temperature_c,omitempty"`
	HeartRateBPM    *int     `json:"heart_rate_bpm,omitempty"`
	RespiratoryRate *int     `json:"respiratory_rate,omitempty"`
	SystolicBP      *int     `json:"systolic_bp,omitempty"`
	DiastolicBP     *int     `json:"diastolic_bp,omitempty"`
}

type DiseasePrediction struct {
	Disease     string  `json:"disease"`
	Probability float64 `json:"probability"`
}

type DiagnosisResponse struct {
	PredictedDisease string              `json:"predicted_disease"`
	Confidence       float64             `json:"confidence"`
	TopCandidates    []DiseasePrediction `json:"top_candidates"`
	Description      string              `json:"description,omitempty"`
	Precautions      []string            `json:"precautions,omitempty"`
	ModelVersion     string              `json:"model_version"`
}

// SymptomsResponse mirrors the Python service's /symptoms response —
// the ordered vocabulary the currently-loaded model was trained on.
type SymptomsResponse struct {
	Symptoms []string `json:"symptoms"`
	Count    int      `json:"count"`
}

type ErrorResponse struct {
	Error  string `json:"error"`
	Detail string `json:"detail,omitempty"`
}
