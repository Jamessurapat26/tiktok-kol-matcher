// Wire types for the HTTP API. Field names mirror the backend models exactly:
// backend/app/api/main.py, store.py, settings.py, and backend/app/models.py.

/** settings.py `Mode` */
export type Mode = 'demo' | 'live'

/** settings.py `LiveFlags` */
export interface LiveFlags {
  live_web: boolean
  live_fb: boolean
  live_kols: boolean
  live_llm: boolean
  live_embed: boolean
}

export type LiveFlag = keyof LiveFlags

/** main.py `Health` */
export interface Health {
  status: 'ok'
  mode: Mode
  defaults: LiveFlags
  credentials: Record<string, boolean>
  models: Record<string, string | null>
}

/** main.py `JobCreate`. An omitted or null flag takes the mode's default. */
export interface JobCreate extends Partial<Record<LiveFlag, boolean | null>> {
  website_url: string
  facebook_url: string
  top_k?: number
}

/** main.py `JobCreated` */
export interface JobCreated {
  job_id: string
}

/** models.py `MatchWeights` */
export interface MatchWeights {
  relevance: number
  engagement: number
  audience: number
  tier: number
}

/** models.py `MatchRequest` */
export interface MatchRequest {
  website_url: string
  facebook_url: string
  top_k: number
  weights: MatchWeights
}

/** models.py `TargetAudience` */
export interface TargetAudience {
  age_min: number | null
  age_max: number | null
  gender: 'male' | 'female' | 'any'
  interests: string[]
  location: string
}

export type Tier = 'nano' | 'micro' | 'macro' | 'mega'

/** models.py `BrandProfile` */
export interface BrandProfile {
  category: string
  products: string[]
  target_audience: TargetAudience
  tone: string
  content_style: string
  keywords_th: string[]
  keywords_en: string[]
  hashtags: string[]
  preferred_tiers: Tier[]
  reasoning: string
}

/** models.py `KolProfile`. The API has no avatar field. */
export interface KolProfile {
  handle: string
  url: string
  nickname: string | null
  followers: number
  bio: string | null
  language: string | null
  region: string | null
  /** Fraction, 0.043 = 4.3%. */
  median_engagement_rate: number | null
  posts_per_week: number
  content_summary: string | null
  source: 'apify' | 'seed' | 'cache'
}

export type AudienceConfidence = 'low' | 'medium' | 'high'

/** models.py `ScoreBreakdown`. Each score is 0..1. */
export interface ScoreBreakdown {
  relevance: number
  engagement: number
  audience_fit: number
  tier_fit: number
  audience_confidence: AudienceConfidence
}

/** models.py `RankedKol` */
export interface RankedKol {
  kol: KolProfile
  scores: ScoreBreakdown
  total_score: number
  reason: string
}

/** store.py `JobStatus` */
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed'

/** store.py `Job` */
export interface Job {
  id: string
  status: JobStatus
  stage: string
  request: MatchRequest
  flags: LiveFlags
  errors: string[]
  brand_profile: BrandProfile | null
  results: RankedKol[]
  created_at: string
  updated_at: string
}

/** store.py `NodeStatus` */
export type NodeStatus = 'started' | 'finished' | 'failed'

export const NODE_STATUSES = ['started', 'finished', 'failed'] as const satisfies readonly NodeStatus[]

interface EventBase {
  job_id: string
  seq: number
  summary: string
  timestamp: string
}

/** store.py `Event` with type "node". The SSE event name equals `type`. */
export interface NodeEvent extends EventBase {
  type: 'node'
  node: string
  status: NodeStatus
}

/** store.py `Event` with type "done" or "failed": the last event of a job. */
export interface TerminalEvent extends EventBase {
  type: 'done' | 'failed'
  node: null
  status: null
}

export type JobEvent = NodeEvent | TerminalEvent

export const EVENT_TYPES = ['node', 'done', 'failed'] as const satisfies readonly JobEvent['type'][]

/** One entry of a FastAPI 422 body. `loc` is e.g. ["body", "website_url"]. */
export interface ValidationIssue {
  type: string
  loc: (string | number)[]
  msg: string
}

/** Error bodies: 422 carries issues, 400/404/429 carry a message. */
export type ErrorBody = { detail: ValidationIssue[] } | { detail: string }
