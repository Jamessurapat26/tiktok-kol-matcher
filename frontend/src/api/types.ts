// Wire types for the HTTP API. Field names mirror the backend models exactly:
// backend/app/api/main.py, store.py, settings.py, backend/app/models.py, and
// backend/app/state.py.

/** settings.py `Mode` */
export type Mode = 'demo' | 'live'

/** settings.py `LiveFlags` */
export interface LiveFlags {
  live_web: boolean
  live_fb: boolean
  live_kols: boolean
  live_llm: boolean
  live_embed: boolean
  live_search: boolean
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

/** models.py `MatchRequest` */
export interface MatchRequest {
  website_url: string
  facebook_url: string
  top_k: number
}

/** models.py `TargetAudience` */
export interface TargetAudience {
  gender: 'male' | 'female' | 'any'
  age_range: string
  interests: string[]
}

export type Tier = 'nano' | 'micro' | 'mid' | 'macro'

/** models.py `ContentFormat` */
export type ContentFormat =
  | 'review'
  | 'tutorial'
  | 'unboxing'
  | 'comparison'
  | 'vlog'
  | 'comedy'
  | 'storytelling'
  | 'challenge'
  | 'lifestyle'
  | 'news'

/** models.py `ContentAnalysis`. For the Marketer only: it moves no score. */
export interface ContentAnalysis {
  formats: ContentFormat[]
  style: string
  style_fit: number | null
  style_reason: string
  evidence_urls: string[]
}

/** models.py `BrandProfile`: the structured brief. */
export interface BrandProfile {
  category: string
  key_products: string[]
  price_tier: 'budget' | 'mid' | 'premium' | 'luxury'
  target_audience: TargetAudience
  tone: string
  /** Absent on briefs stored before the fields existed. */
  content_formats?: ContentFormat[]
  content_style?: string
  topic_keywords_th: string[]
  topic_keywords_en: string[]
  competitors: string[]
  /** Stores that sell the brand. Absent on briefs stored before the field existed. */
  retailers?: string[]
  brand_handles: string[]
  brand_names: string[]
  target_followers: number
  tier_label: Tier
  tier_confidence: number
  tier_reason: string
  /** Thai display text. Absent on briefs stored before the fields existed. */
  category_th?: string
  key_products_th?: string[]
  tone_th?: string
  content_style_th?: string
  interests_th?: string[]
  tier_reason_th?: string
}

/** models.py `VideoStats` */
export interface VideoStats {
  url: string | null
  views: number
  likes: number
  comments: number
  shares: number
  posted_at: string | null
  caption: string
  /** Absent on rows stored before the fields existed. */
  hashtags?: string[]
  is_ad?: boolean
}

/** models.py `KolProfile`. The API has no avatar field. */
export interface KolProfile {
  handle: string
  url: string
  nickname: string | null
  /** TikTok profile picture URL. Absent on rows stored before the field existed. */
  avatar_url?: string | null
  followers: number
  bio: string | null
  language: string | null
  region: string | null
  posts_per_week: number
  content_summary: string | null
  videos: VideoStats[]
  stats_source: 'search' | 'profile'
  source: 'apify' | 'seed' | 'cache'
}

/** models.py `Scores`. Each score and confidence is 0..1. */
export interface Scores {
  F: number
  E: number
  T: number
  A_eff: number
  F_conf: number
  E_conf: number
  T_conf: number
  A_conf: number
  sim: number
  on_topic: number
  judge: number
  er_star: number | null
  reach: number | null
  judged: boolean
}

/** models.py `RankedKol` */
export interface RankedKol {
  kol: KolProfile
  scores: Scores
  quality: number
  final: number
  flags: string[]
  reason: string
  evidence_urls: string[]
  profile_url: string
  /** Only for judged KOLs. Absent on rows stored before the field existed. */
  content?: ContentAnalysis | null
}

/** state.py `MatchState.stage`, in pipeline order. */
export type Stage =
  | 'received'
  | 'scraping'
  | 'analyzing_brand'
  | 'generating_queries'
  | 'discovering_kols'
  | 'filtering'
  | 'first_pass'
  | 'enriching_kols'
  | 'judging'
  | 'outputting'
  | 'done'
  | 'failed'

/** store.py `JobStatus` */
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed'

/** store.py `Job` */
export interface Job {
  id: string
  status: JobStatus
  stage: Stage
  request: MatchRequest
  flags: LiveFlags
  errors: string[]
  brand_profile: BrandProfile | null
  /** Facebook page picture. Absent on jobs stored before the field existed. */
  brand_logo_url?: string | null
  rejected: Record<string, number>
  cost_log: string[]
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
