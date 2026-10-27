import demoEvents from '@/test/demo-events.json'
import type { Job } from './types'
import { ApiError, api } from './client'
import { followJobEvents, parseJobEvent } from './events'

vi.mock('./client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./client')>()
  return { ...actual, api: { ...actual.api, job: vi.fn() } }
})

class FakeEventSource {
  static readonly CLOSED = 2
  static readonly instances: FakeEventSource[] = []
  static get last(): FakeEventSource {
    const source = FakeEventSource.instances.at(-1)
    if (!source) throw new Error('no EventSource opened')
    return source
  }

  readonly url: string
  readyState = 0
  closed = false
  onopen: (() => void) | null = null
  onerror: (() => void) | null = null
  private listeners = new Map<string, (message: MessageEvent<string>) => void>()

  constructor(url: string) {
    this.url = url
    FakeEventSource.instances.push(this)
  }

  addEventListener(type: string, listener: (message: MessageEvent<string>) => void) {
    this.listeners.set(type, listener)
  }

  close() {
    this.closed = true
  }
}

function handlers() {
  return {
    onEvent: vi.fn(),
    onConnectionChange: vi.fn(),
    onReopen: vi.fn(),
    onGone: vi.fn(),
  }
}

beforeEach(() => {
  FakeEventSource.instances.length = 0
  vi.mocked(api.job).mockReset()
  vi.stubGlobal('EventSource', FakeEventSource)
  vi.useFakeTimers()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('followJobEvents', () => {
  it('stops retrying once a probe reports the job is gone', async () => {
    vi.mocked(api.job).mockRejectedValue(new ApiError(404, { detail: 'Job not found' }))
    const h = handlers()
    followJobEvents('missing', h)
    const source = FakeEventSource.last

    source.readyState = FakeEventSource.CLOSED
    source.onerror?.()
    await vi.advanceTimersByTimeAsync(0)

    expect(api.job).toHaveBeenCalledWith('missing')
    expect(h.onGone).toHaveBeenCalledWith(expect.objectContaining({ status: 404 }))
    expect(source.closed).toBe(true)

    await vi.advanceTimersByTimeAsync(10_000)
    expect(FakeEventSource.instances).toHaveLength(1)
  })

  it('treats a 429 probe as transient and reopens', async () => {
    vi.mocked(api.job).mockRejectedValue(new ApiError(429, { detail: 'Too many requests' }))
    const h = handlers()
    followJobEvents('busy', h)
    const source = FakeEventSource.last

    source.readyState = FakeEventSource.CLOSED
    source.onerror?.()
    await vi.advanceTimersByTimeAsync(0)

    expect(h.onGone).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(1000)
    expect(FakeEventSource.instances).toHaveLength(2)
    expect(h.onReopen).toHaveBeenCalledOnce()
  })

  it('reopens after a transient probe failure such as a 503', async () => {
    vi.mocked(api.job).mockRejectedValue(new ApiError(503, null))
    const h = handlers()
    followJobEvents('restarting', h)
    const source = FakeEventSource.last

    source.readyState = FakeEventSource.CLOSED
    source.onerror?.()
    await vi.advanceTimersByTimeAsync(0)

    expect(h.onGone).not.toHaveBeenCalled()
    expect(FakeEventSource.instances).toHaveLength(1)
    await vi.advanceTimersByTimeAsync(1000)
    expect(FakeEventSource.instances).toHaveLength(2)
    expect(h.onReopen).toHaveBeenCalledOnce()
  })

  it('does not reopen when stopped while the probe is in flight', async () => {
    let finishProbe: (job: Job) => void = () => {}
    vi.mocked(api.job).mockImplementation(() => new Promise<Job>((resolve) => (finishProbe = resolve)))
    const h = handlers()
    const stop = followJobEvents('slow', h)
    const source = FakeEventSource.last

    source.readyState = FakeEventSource.CLOSED
    source.onerror?.()
    stop()
    finishProbe({} as Job)
    await vi.advanceTimersByTimeAsync(10_000)

    expect(h.onGone).not.toHaveBeenCalled()
    expect(FakeEventSource.instances).toHaveLength(1)
  })
})

describe('parseJobEvent', () => {
  it('accepts a real node event', () => {
    expect(parseJobEvent(JSON.stringify(demoEvents[0]))).toEqual(demoEvents[0])
  })

  it('rejects a node event whose status is not one the store emits', () => {
    expect(parseJobEvent(JSON.stringify({ ...demoEvents[0], status: 'paused' }))).toBeNull()
    expect(parseJobEvent(JSON.stringify({ ...demoEvents[0], status: null }))).toBeNull()
  })

  it('rejects payloads that are not job events', () => {
    expect(parseJobEvent('not json')).toBeNull()
    expect(parseJobEvent('{"seq":1,"type":"ping"}')).toBeNull()
  })
})
