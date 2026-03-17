export interface AnalysisResult {
  image_id: string
  quality: { score: number; adequate: boolean }
  dr_grading: {
    grade: 0|1|2|3|4
    label: string
    confidence: number
    probabilities: number[]
    refer: boolean
  }
  amd:      { stage: number; label: string; confidence: number }
  glaucoma: { suspect: boolean; cup_disc_ratio: number; confidence: number }
  lesions:  Record<string, { present: boolean; probability: number }>
  report:   { structured_findings: string; recommendation: string; full_text: string }
  explainability: {
    gradcam_image:     string | null
    attention_image:   string | null
    explanation_panel: string | null
  }
  segmentation: {
    vessel_mask:     string | null
    optic_disc_mask: string | null
  }
  inference_time_ms: number
  model_version?: string
}

export interface CaseStats {
  total_cases: number
  today:       number
  this_week:   number
  referable_cases: number
  dr_grade_distribution: Record<string, number>
}

export interface CaseEntry {
  id: string
  patient_id: string
  created_at: string
  image_name: string
  dr_grade:   number
  dr_label:   string
  dr_confidence: number
  dr_refer:   number
  quality_score: number
  quality_adequate: number
  status: string
}

export interface SearchResult {
  rank:       number
  image_id:   string
  score:      number
  distance:   number
  dr_grade:   number | null
  dr_label:   string | null
  dataset:    string | null
  image_path: string | null
}

export interface ProgressionReport {
  patient_id:    string
  num_visits:    number
  visit_dates:   string[]
  overall_trend: string
  dr_grades:     number[]
  grade_change:  number
  risk_12m:      number
  risk_level:    string
  new_lesions:   string[]
  recommendation: string
  full_report:   string
}

export const DR_LABEL: Record<number, string> = {
  0: 'No DR', 1: 'Mild NPDR', 2: 'Moderate NPDR', 3: 'Severe NPDR', 4: 'Proliferative DR'
}
