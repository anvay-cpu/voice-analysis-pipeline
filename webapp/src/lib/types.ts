// ── Core Report Data Types ──────────────────────────────────────────

export interface DimensionScores {
  vocal_clarity: number;
  body_language: number;
  content_structure: number;
  audience_engagement: number;
  emotional_expressiveness: number;
  regime_adaptability: number;
  overall: number;
}

export interface ContentDetails {
  grammar_score: number;
  fkgl: number;
  ttr: number;
  dominant_tone: string;
  dominant_sentiment: string;
  argument_effectiveness: number;
}

export interface FusionStats {
  timeline_duration: number;
  regime_boundaries: number;
  total_disruptions: number;
  mean_composure: number;
  emotion_coherence: number;
}

export interface RegimeSegment {
  regime_type: string;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
}

export interface RegimeTransition {
  time_sec: number;
  from_regime: string;
  to_regime: string;
  score: number;
}

export interface DisruptionEvent {
  time_sec: number;
  timestamp: string;
  type: "disfluency" | "emotion_mismatch" | "transition" | "filler_burst";
  label: string;
  composure: number | null;
  recovery_sec: number | null;
  status: "GOOD" | "SLOW" | "MISMATCH" | string;
}

export interface CoachingBlock {
  time_sec: number;
  timestamp: string;
  type: string;
  label: string;
  feedback: string;
}

export interface PracticeExercise {
  title: string;
  description: string;
  frequency: string;
  target_dimension?: string;
  target_improvement?: number;
}

export interface FillerWord {
  time_sec: number;
  timestamp: string;
  word: string;
  context: string;
}

export interface TimelinePoint {
  time_sec: number;
  voice: {
    syllable_rate: number;
    pitch_cv: number;
    emotion_label: string;
    filler_count: number;
    disfluency_count: number;
  };
  body: {
    posture_score: number;
    gesture_dominant: string;
    gaze_engagement: number;
    facial_emotion: string;
  };
  content: {
    regime_type: string;
    grammar_error_count: number;
  };
}

export interface EmotionArcPoint {
  time_sec: number;
  voice_valence: number;
  face_valence: number;
  text_valence: number;
}

export interface HeatmapChannel {
  label: string;
  key: string;
  values: number[]; // 0-1 quality per second
}

export interface SpeechReport {
  id: string;
  title: string;
  duration_sec: number;
  processing_time_sec: number;
  video_url?: string;
  scores: DimensionScores;
  content_details: ContentDetails;
  fusion_stats: FusionStats;
  executive_summary: string;
  regime_segments: RegimeSegment[];
  regime_transitions: RegimeTransition[];
  disruption_events: DisruptionEvent[];
  coaching_feedback: CoachingBlock[];
  practice_plan: PracticeExercise[];
  filler_words: FillerWord[];
  transcript: string;
  timeline: TimelinePoint[];
  emotion_arc: EmotionArcPoint[];
  heatmap_channels: HeatmapChannel[];
}

export type ViewType = "dashboard" | "summary" | "timeline" | "coaching" | "data" | "video" | "practice";

export type TopNavMode = "LIVE_FEED" | "ANALYSIS" | "HISTORY" | "SETTINGS";

// ── Backend API Types ──────────────────────────────────────────────

export interface UploadResponse {
  job_id: string;
  session_id: string;
  status: string;
  message: string;
}

export interface JobStatus {
  job_id: string;
  session_id: string;
  status: "queued" | "running" | "complete" | "failed";
  step: string;
  step_index: number;
  total_steps: number;
  progress: number;
  error?: string;
  report_id?: string;
  processing_time_sec?: number;
}

export interface SessionEntry {
  id: string;
  title: string;
  profile: string;
  filename: string;
  file_size: number;
  status: "PROCESSING" | "COMPLETE" | "FAILED";
  score: number | null;
  dominant_tone: string | null;
  created_at: string;
  duration: string | null;
  processing_time_sec: number | null;
}
