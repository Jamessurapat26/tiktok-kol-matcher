import type { JobEvent } from '@/api/types'
import demoEvents from '@/test/demo-events.json'
import { applyEvent, initialProgress, NODE_ORDER, type Progress, stepDuration } from './steps'

const events = demoEvents as JobEvent[]
const fold = (list: JobEvent[], from: Progress = initialProgress()) => list.reduce(applyEvent, from)
function step(progress: Progress, node: string) {
  const found = progress.steps.find((s) => s.node === node)
  if (!found) throw new Error(`no step ${node}`)
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
  it('starts with every graph node pending, in graph order', () => {
    const progress = initialProgress()
    expect(progress.steps.map((s) => s.node)).toEqual([...NODE_ORDER])
    expect(progress.steps.every((s) => s.status === 'pending')).toBe(true)
  })

  it('turns a full demo stream into finished steps with summaries and durations', () => {
    const progress = fold(events)
    expect(progress.steps.every((s) => s.status === 'finished')).toBe(true)
    expect(step(progress, 'filter_thai').summary).toBe('candidates=25')
    expect(step(progress, 'scrape_website').summary).toBe('website=1330 chars')
    expect(stepDuration(step(progress, 'analyze_brand'))).toBeGreaterThanOrEqual(0)
    expect(progress.outcome).toEqual({ type: 'done', summary: 'ranked 3 KOLs' })
    expect(progress.lastSeq).toBe(25)
  })

  it('shows a node as running between its started and finished events', () => {
    const progress = fold(events.slice(0, 9))
    expect(step(progress, 'discover_kols').status).toBe('running')
    expect(step(progress, 'filter_thai').status).toBe('pending')
    expect(stepDuration(step(progress, 'discover_kols'))).toBeNull()
  })

  it('counts the discovery loop as a second round and keeps the latest summary', () => {
    const afterFirstRound = fold(events.slice(0, 10))
    expect(step(afterFirstRound, 'discover_kols')).toMatchObject({ status: 'finished', runs: 1 })

    const duringSecondRound = fold(events.slice(10, 13), afterFirstRound)
    expect(step(duringSecondRound, 'discover_kols')).toMatchObject({ status: 'running', runs: 2 })
    expect(step(duringSecondRound, 'generate_queries').runs).toBe(2)

    const afterSecondRound = fold(events.slice(13, 14), duringSecondRound)
    expect(step(afterSecondRound, 'discover_kols')).toMatchObject({
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
    expect(step(replayed, 'discover_kols').runs).toBe(2)
    expect(applyEvent(replayed, events[3])).toBe(replayed)
  })

  it('marks a failed node red with its message and stops there', () => {
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
    expect(step(progress, 'analyze_brand')).toMatchObject({
      status: 'failed',
      error: 'analyze_brand: model unavailable',
    })
    expect(step(progress, 'generate_queries').status).toBe('pending')
    expect(progress.outcome).toEqual({ type: 'failed', summary: 'model unavailable' })
  })

  it('fails a still-running node when the job fails without a node event', () => {
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
    expect(step(progress, 'scrape_website')).toMatchObject({
      status: 'failed',
      error: 'interrupted by restart',
    })
  })

  it('skips nodes it does not know', () => {
    const progress = fold([nodeEvent(1, 'new_node', 'started')])
    expect(progress.steps).toEqual(initialProgress().steps)
    expect(progress.lastSeq).toBe(1)
  })
})
