import type { JobEvent, Stage } from '@/api/types'

/**
 * backend/app/state.py `MatchState.stage`, in pipeline order.
 *
 * Node events still name graph nodes; `NODE_STAGE` folds them into these
 * user-facing stages (the two scrapers share "scraping").
 */
export const STAGE_ORDER = [
  'received',
  'scraping',
  'analyzing_brand',
  'generating_queries',
  'discovering_kols',
  'filtering',
  'first_pass',
  'enriching_kols',
  'judging',
  'outputting',
  'done',
] as const

export type StageName = (typeof STAGE_ORDER)[number]

export const STAGE_LABELS: Record<StageName, string> = {
  received: 'Request received',
  scraping: 'Read brand sources',
  analyzing_brand: 'Analyze brand',
  generating_queries: 'Generate search queries',
  discovering_kols: 'Discover KOLs',
  filtering: 'Filter creators',
  first_pass: 'First-pass ranking',
  enriching_kols: 'Enrich profiles',
  judging: 'Judge and rank',
  outputting: 'Rank and output',
  done: 'Done',
}

/** backend/app/graph.py `_NODE_ORDER` mapped onto the stage list above. */
const NODE_STAGE: Record<string, StageName> = {
  scrape_website: 'scraping',
  scrape_facebook: 'scraping',
  analyze_brand: 'analyzing_brand',
  generate_queries: 'generating_queries',
  discover_kols: 'discovering_kols',
  hard_filter: 'filtering',
  first_pass_rank: 'first_pass',
  enrich_kols: 'enriching_kols',
  judge_and_rank: 'judging',
  rank_and_output: 'outputting',
}

export type StepStatus = 'pending' | 'running' | 'finished' | 'failed'

export interface Step {
  stage: StageName
  status: StepStatus
  /** Summary of the latest run. */
  summary: string
  /** Start of the latest run. */
  startedAt: string | null
  /** End of the latest run. */
  finishedAt: string | null
  error: string | null
  /** How many times the stage started. The discovery loop runs some stages twice. */
  runs: number
  /** Nodes of this stage that started and have not ended yet. */
  activeNodes: string[]
}

export interface Progress {
  steps: Step[]
  /** Highest seq applied. Events at or below it are replays and are ignored. */
  lastSeq: number
  outcome: { type: 'done' | 'failed'; summary: string } | null
}

export function initialProgress(): Progress {
  return {
    steps: STAGE_ORDER.map((stage) => ({
      stage,
      status: 'pending',
      summary: '',
      startedAt: null,
      finishedAt: null,
      error: null,
      runs: 0,
      activeNodes: [],
    })),
    lastSeq: 0,
    outcome: null,
  }
}

function stageForNode(node: string): StageName | null {
  return NODE_STAGE[node] ?? null
}

function updateStep(progress: Progress, stage: StageName, change: (step: Step) => Step): Step[] {
  return progress.steps.map((step) => (step.stage === stage ? change(step) : step))
}

/** The request has been received as soon as the first node event arrives. */
function finishReceived(steps: Step[], at: string): Step[] {
  return steps.map((step) =>
    step.stage === 'received' && step.status === 'pending'
      ? { ...step, status: 'finished', startedAt: at, finishedAt: at }
      : step,
  )
}

/** Fold one SSE event into the progress. Pure, and idempotent per seq. */
export function applyEvent(progress: Progress, event: JobEvent): Progress {
  if (event.seq <= progress.lastSeq) return progress
  const next = { ...progress, lastSeq: event.seq }

  if (event.type !== 'node') {
    if (event.type === 'failed') {
      // A job can fail without a node failure (e.g. "interrupted by restart"),
      // so whatever was still running ends failed with the job's reason.
      const steps = progress.steps.map((step) =>
        step.status === 'running'
          ? {
              ...step,
              status: 'failed' as const,
              error: event.summary,
              finishedAt: event.timestamp,
              activeNodes: [],
            }
          : step,
      )
      return { ...next, steps, outcome: { type: event.type, summary: event.summary } }
    }
    const steps = finishReceived(progress.steps, event.timestamp).map((step) =>
      step.stage === 'done' ? { ...step, status: 'finished' as const, finishedAt: event.timestamp } : step,
    )
    return { ...next, steps, outcome: { type: event.type, summary: event.summary } }
  }

  const stage = stageForNode(event.node)
  if (!stage) return next
  const at = event.timestamp
  const steps = finishReceived(progress.steps, at)

  const node = event.node
  switch (event.status) {
    case 'started':
      return {
        ...next,
        steps: updateStep({ ...progress, steps }, stage, (step) =>
          step.activeNodes.length > 0
            ? // The two scrapers share one stage; a second start joins the current run.
              { ...step, activeNodes: [...step.activeNodes, node] }
            : {
                ...step,
                status: 'running',
                summary: '',
                startedAt: at,
                finishedAt: null,
                error: null,
                runs: step.runs + 1,
                activeNodes: [node],
              },
        ),
      }
    case 'finished':
    case 'failed': {
      const failed = event.status === 'failed'
      return {
        ...next,
        steps: updateStep({ ...progress, steps }, stage, (step) => {
          const activeNodes = step.activeNodes.filter((name) => name !== node)
          const summary = failed ? step.summary : joinSummary(step.summary, event.summary)
          const error = failed ? joinSummary(step.error ?? '', event.summary) : step.error
          // A shared stage ends only when its last node ends, and a failure sticks.
          const status =
            failed || step.status === 'failed' ? 'failed' : activeNodes.length > 0 ? 'running' : 'finished'
          return {
            ...step,
            status,
            summary,
            error,
            startedAt: step.startedAt ?? at,
            finishedAt: activeNodes.length > 0 ? null : at,
            activeNodes,
          }
        }),
      }
    }
    default:
      return progress
  }
}

function joinSummary(current: string, addition: string): string {
  if (!current) return addition
  if (!addition) return current
  return `${current} · ${addition}`
}

/** Milliseconds of the latest run, or null until it ends. */
export function stepDuration(step: Step): number | null {
  if (!step.startedAt || !step.finishedAt) return null
  return Math.max(0, Date.parse(step.finishedAt) - Date.parse(step.startedAt))
}

/** A stage from the wire enum, narrowed to one of the visible steps. */
export function isStage(value: Stage): value is StageName {
  return (STAGE_ORDER as readonly string[]).includes(value)
}
