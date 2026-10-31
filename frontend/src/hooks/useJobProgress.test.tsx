import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, render, screen, within } from '@testing-library/react'
import { ApiError, api } from '@/api/client'
import type { JobEvent } from '@/api/types'
import { ProgressCard } from '@/components/ProgressCard'
import demoEvents from '@/test/demo-events.json'
import { useJobProgress } from './useJobProgress'

const events = demoEvents as JobEvent[]

class FakeEventSource {
  static readonly CLOSED = 2
  static last: FakeEventSource
  readonly url: string
  readyState = 0
  closed = false
  onopen: (() => void) | null = null
  onerror: (() => void) | null = null
  private listeners = new Map<string, (message: MessageEvent<string>) => void>()

  constructor(url: string) {
    this.url = url
    FakeEventSource.last = this
  }

  addEventListener(type: string, listener: (message: MessageEvent<string>) => void) {
    this.listeners.set(type, listener)
  }

  close() {
    this.closed = true
  }

  emit(list: JobEvent[]) {
    act(() => {
      for (const event of list) {
        this.listeners.get(event.type)?.(
          new MessageEvent(event.type, { data: JSON.stringify(event), lastEventId: String(event.seq) }),
        )
      }
    })
  }
}

function Harness({ jobId }: { jobId: string }) {
  const { progress, connected } = useJobProgress(jobId)
  return <ProgressCard progress={progress} connected={connected} />
}

function renderHarness() {
  const client = new QueryClient()
  const invalidate = vi.spyOn(client, 'invalidateQueries')
  render(
    <QueryClientProvider client={client}>
      <Harness jobId={events[0].job_id} />
    </QueryClientProvider>,
  )
  return { source: FakeEventSource.last, invalidate }
}

function row(name: string) {
  const item = screen.getByText(name).closest('li')
  if (!item) throw new Error(`no row ${name}`)
  return within(item)
}

beforeEach(() => vi.stubGlobal('EventSource', FakeEventSource))
afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('job progress over SSE', () => {
  it('subscribes to the job stream and fills the stepper as events arrive', () => {
    const { source, invalidate } = renderHarness()
    expect(source.url).toBe(`/api/jobs/${events[0].job_id}/events`)
    act(() => source.onopen?.())
    expect(screen.getByText('กำลังเริ่มการค้นหา')).toBeInTheDocument()

    source.emit(events.slice(0, 8))
    expect(row('วิเคราะห์แบรนด์').getByText(', เสร็จแล้ว')).toBeInTheDocument()
    expect(row('ค้นหา KOL บน TikTok').getByText(', กำลังทำงาน')).toBeInTheDocument()
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ['jobs', events[0].job_id] })
    expect(row('คัดกรองและประเมิน KOL').getByText(', รอดำเนินการ')).toBeInTheDocument()
  })

  it('keeps the discovery loop inside one step and hides its internals', () => {
    const { source } = renderHarness()
    source.emit(events.slice(0, 13))
    expect(row('ค้นหา KOL บน TikTok').getByText(', กำลังทำงาน')).toBeInTheDocument()
    expect(screen.getByText('กำลังค้นหา KOL บน TikTok')).toBeInTheDocument()
    expect(screen.queryByText(/round/)).not.toBeInTheDocument()
    expect(screen.queryByText(/queries\+/)).not.toBeInTheDocument()
  })

  it('survives a dropped connection whose replay overlaps, then closes on done', () => {
    const { source } = renderHarness()
    source.emit(events.slice(0, 12))
    act(() => source.onerror?.())
    // The server resumes after Last-Event-ID, but a replay may still overlap.
    source.emit(events.slice(9))

    expect(screen.getByText('จัดอันดับ KOL เสร็จแล้ว')).toBeInTheDocument()
    expect(screen.getAllByText(', เสร็จแล้ว')).toHaveLength(5)
    expect(source.closed).toBe(true)
  })

  it('reopens after the last seq when the browser gives up on the stream', async () => {
    vi.useFakeTimers()
    vi.spyOn(api, 'job').mockRejectedValue(new ApiError(503, null))
    const { source, invalidate } = renderHarness()
    source.emit(events.slice(0, 12))
    invalidate.mockClear()

    // A non-200 reconnect response (a 502 while the backend restarts) closes
    // the EventSource for good instead of retrying.
    await act(async () => {
      source.readyState = FakeEventSource.CLOSED
      source.onerror?.()
    })
    expect(FakeEventSource.last).toBe(source)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })

    const reopened = FakeEventSource.last
    expect(reopened).not.toBe(source)
    expect(reopened.url).toBe(`/api/jobs/${events[0].job_id}/events?after=12`)
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ['jobs', events[0].job_id] })

    reopened.emit(events.slice(9))
    expect(screen.getByText('จัดอันดับ KOL เสร็จแล้ว')).toBeInTheDocument()
    expect(screen.getAllByText(', เสร็จแล้ว')).toHaveLength(5)
    expect(reopened.closed).toBe(true)
  })
})
