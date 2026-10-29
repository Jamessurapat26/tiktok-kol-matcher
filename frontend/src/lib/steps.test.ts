import type { JobEvent, NodeStatus } from '@/api/types'
import demoEvents from '@/test/demo-events.json'
import { applyEvent, initialProgress, type Progress, STAGE_LABELS, STAGE_ORDER, stepDuration } from './steps'

const events = demoEvents as JobEvent[]
const fold = (list: JobEvent[], from: Progress = initialProgress()) => list.reduce(applyEvent, from)
function step(progress: Progress, stage: string) {
  const found = progress.steps.find((s) => s.stage === stage)
  if (!found) throw new Error(`no step ${stage}`)
  return found
}

function nodeEvent(
  seq: number,
  node: string,
  status: 'started' | 'finished' | 'failed',
  summary = '',
): JobEvent {
  return {
    job_id: 'job',
    seq,
    type: 'node',
    node,
    status,
    summary,
    timestamp: new Date(Date.UTC(2026, 9, 27, 0, 0, seq)).toISOString(),
  }
}

describe('applyEvent', () => {
  it('starts with every pipeline stage pending, in backend order', () => {
    const progress = initialProgress()
    expect(STAGE_ORDER).toEqual([
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
    ])
    expect(progress.steps.map((s) => s.stage)).toEqual([...STAGE_ORDER])
    expect(progress.steps.every((s) => s.status === 'pending')).toBe(true)
  })

  it('gives every stage a readable label', () => {
    for (const stage of STAGE_ORDER) {
      expect(STAGE_LABELS[stage]).toBeTruthy()
      expect(STAGE_LABELS[stage]).not.toBe(stage)
    }
  })

  it('turns a full demo stream into finished stages with summaries and durations', () => {
    const progress = fold(events)
    expect(progress.steps.every((s) => s.status === 'finished')).toBe(true)
    expect(step(progress, 'filtering').summary).toBe('candidates=25')
    expect(step(progress, 'scraping').summary).toBe('facebook=ExampleBrand')
    expect(stepDuration(step(progress, 'analyzing_brand'))).toBeGreaterThanOrEqual(0)
    expect(progress.outcome).toEqual({ type: 'done', summary: 'ranked 3 KOLs' })
    expect(progress.lastSeq).toBe(25)
  })

  it('folds both scrapers into one scraping stage', () => {
    const progress = fold(events.slice(0, 4))
    expect(step(progress, 'scraping')).toMatchObject({ status: 'finished', runs: 1 })
    expect(step(progress, 'received').status).toBe('finished')
  })

  it('shows a stage as running between its started and finished events', () => {
    const progress = fold(events.slice(0, 9))
    expect(step(progress, 'discovering_kols').status).toBe('running')
    expect(step(progress, 'filtering').status).toBe('pending')
    expect(stepDuration(step(progress, 'discovering_kols'))).toBeNull()
  })

  it('counts the discovery loop as a second round and keeps the latest summary', () => {
    const afterFirstRound = fold(events.slice(0, 10))
    expect(step(afterFirstRound, 'discovering_kols')).toMatchObject({ status: 'finished', runs: 1 })

    const duringSecondRound = fold(events.slice(10, 13), afterFirstRound)
    expect(step(duringSecondRound, 'discovering_kols')).toMatchObject({
      status: 'running',
      runs: 2,
      summary: '',
    })
    expect(step(duringSecondRound, 'generating_queries').runs).toBe(2)

    const afterSecondRound = fold(events.slice(13, 14), duringSecondRound)
    expect(step(afterSecondRound, 'discovering_kols')).toMatchObject({
      status: 'finished',
      runs: 2,
      summary: 'round=2 discovered+18',
    })
  })

  it('ignores events replayed after a reconnect', () => {
    const live = fold(events.slice(0, 12))
    // A reconnect that resends events the client already has, then continues.
    const replayed = fold([...events.slice(8, 12), ...events.slice(12)], live)
    expect(replayed).toEqual(fold(events))
    expect(step(replayed, 'discovering_kols').runs).toBe(2)
    expect(applyEvent(replayed, events[3])).toBe(replayed)
  })

  it('marks a failed stage red with its message and stops there', () => {
    const progress = fold([
      nodeEvent(1, 'scrape_website', 'started'),
      nodeEvent(2, 'scrape_website', 'finished', 'website=10 chars'),
      nodeEvent(3, 'analyze_brand', 'started'),
      nodeEvent(4, 'analyze_brand', 'failed', 'analyze_brand: model unavailable'),
      {
        ...nodeEvent(5, '', 'failed'),
        type: 'failed',
        node: null,
        status: null,
        summary: 'model unavailable',
      },
    ])
    expect(step(progress, 'analyzing_brand')).toMatchObject({
      status: 'failed',
      error: 'analyze_brand: model unavailable',
    })
    expect(step(progress, 'generating_queries').status).toBe('pending')
    expect(progress.outcome).toEqual({ type: 'failed', summary: 'model unavailable' })
  })

  it('fails a still-running stage when the job fails without a node event', () => {
    const progress = fold([
      nodeEvent(1, 'scrape_website', 'started'),
      {
        ...nodeEvent(2, '', 'failed'),
        type: 'failed',
        node: null,
        status: null,
        summary: 'interrupted by restart',
      },
    ])
    expect(step(progress, 'scraping')).toMatchObject({
      status: 'failed',
      error: 'interrupted by restart',
    })
  })

  it('skips nodes it does not know', () => {
    const progress = fold([nodeEvent(1, 'new_node', 'started')])
    expect(progress.steps).toEqual(initialProgress().steps)
    expect(progress.lastSeq).toBe(1)
  })

  it('leaves the state untouched on a status it does not know', () => {
    const before = fold(events.slice(0, 3))
    const bogus = nodeEvent(4, 'scrape_facebook', 'paused' as NodeStatus)
    expect(applyEvent(before, bogus)).toBe(before)
  })
})
