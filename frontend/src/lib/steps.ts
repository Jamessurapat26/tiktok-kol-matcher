import type { JobEvent } from '@/api/types'

/** backend/app/graph.py `_NODE_ORDER` */
export const NODE_ORDER = [
  'scrape_website',
  'scrape_facebook',
  'analyze_brand',
  'generate_queries',
  'discover_kols',
  'filter_thai',
  'enrich_kols',
  'embed_and_prefilter',
  'llm_rerank',
  'rank_and_output',
] as const

export type NodeName = (typeof NODE_ORDER)[number]

export type StepStatus = 'pending' | 'running' | 'finished' | 'failed'

export interface Step {
  node: NodeName
  status: StepStatus
  /** Summary of the latest run. */
  summary: string
  /** Start of the latest run. */
  startedAt: string | null
  finishedAt: string | null
  error: string | null
  /** How many times the node started. The discovery loop runs some nodes twice. */
  runs: number
}

export interface Progress {
  steps: Step[]
  /** Highest seq applied. Events at or below it are replays and are ignored. */
  lastSeq: number
  outcome: { type: 'done' | 'failed'; summary: string } | null
}

export function initialProgress(): Progress {
  return {
    steps: NODE_ORDER.map((node) => ({
      node,
      status: 'pending',
      summary: '',
      startedAt: null,
      finishedAt: null,
      error: null,
      runs: 0,
    })),
    lastSeq: 0,
    outcome: null,
  }
}

function isNodeName(node: string): node is NodeName {
  return (NODE_ORDER as readonly string[]).includes(node)
}

function updateStep(progress: Progress, node: NodeName, change: (step: Step) => Step): Step[] {
  return progress.steps.map((step) => (step.node === node ? change(step) : step))
}

/** Fold one SSE event into the progress. Pure, and idempotent per seq. */
export function applyEvent(progress: Progress, event: JobEvent): Progress {
  if (event.seq <= progress.lastSeq) return progress
  const next = { ...progress, lastSeq: event.seq }

  if (event.type !== 'node') {
    // A job can fail without a node failure (e.g. "interrupted by restart"),
    // so whatever was still running ends failed with the job's reason.
    const steps =
      event.type === 'failed'
        ? progress.steps.map((step) =>
            step.status === 'running'
              ? { ...step, status: 'failed' as const, error: event.summary, finishedAt: event.timestamp }
              : step,
          )
        : progress.steps
    return { ...next, steps, outcome: { type: event.type, summary: event.summary } }
  }

  if (!isNodeName(event.node)) return next
  const at = event.timestamp
  switch (event.status) {
    case 'started':
      return {
        ...next,
        steps: updateStep(progress, event.node, (step) => ({
          ...step,
          status: 'running',
          summary: '',
          startedAt: at,
          finishedAt: null,
          error: null,
          runs: step.runs + 1,
        })),
      }
    case 'finished':
      return {
        ...next,
        steps: updateStep(progress, event.node, (step) => ({
          ...step,
          status: 'finished',
          summary: event.summary,
          finishedAt: at,
        })),
      }
    case 'failed':
      return {
        ...next,
        steps: updateStep(progress, event.node, (step) => ({
          ...step,
          status: 'failed',
          error: event.summary,
          finishedAt: at,
        })),
      }
    default:
      return progress
  }
}

/** Milliseconds of the latest run, or null until it ends. */
export function stepDuration(step: Step): number | null {
  if (!step.startedAt || !step.finishedAt) return null
  return Math.max(0, Date.parse(step.finishedAt) - Date.parse(step.startedAt))
}
