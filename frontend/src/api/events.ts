import { ApiError, api } from './client'
import { EVENT_TYPES, type JobEvent, NODE_STATUSES, type NodeStatus } from './types'

/** Parse one SSE `data` payload; null when it is not a job event. */
export function parseJobEvent(data: string): JobEvent | null {
  let value: unknown
  try {
    value = JSON.parse(data)
  } catch {
    return null
  }
  if (typeof value !== 'object' || value === null) return null
  const event = value as Partial<JobEvent>
  if (typeof event.seq !== 'number' || !EVENT_TYPES.includes(event.type as JobEvent['type'])) {
    return null
  }
  if (
    event.type === 'node' &&
    (typeof event.node !== 'string' || !NODE_STATUSES.includes(event.status as NodeStatus))
  ) {
    return null
  }
  return event as JobEvent
}

export interface EventStreamHandlers {
  onEvent: (event: JobEvent) => void
  onConnectionChange: (open: boolean) => void
  /** A fresh EventSource was opened; the job may have changed meanwhile. */
  onReopen: () => void
  /** The job itself is gone, so the stream was closed for good. */
  onGone?: (error: ApiError) => void
}

const MAX_RETRY_MS = 10_000

function isGoneStatus(status: number): boolean {
  return status >= 400 && status < 500 && status !== 408 && status !== 429
}

/**
 * Follow a job's events. EventSource reconnects on its own and sends
 * Last-Event-ID, so the server resumes after the last seq it delivered.
 * A reconnect that gets a non-200 (a 502 while the backend restarts) closes
 * the EventSource for good, so that case probes the job and reopens one after
 * a backoff with `?after=<last seq>` unless the job itself is gone. The stream
 * is closed on the terminal event; otherwise it would reconnect to a finished
 * job forever.
 */
export function followJobEvents(jobId: string, handlers: EventStreamHandlers): () => void {
  let source: EventSource
  let retry: ReturnType<typeof setTimeout> | undefined
  let lastSeq = 0
  let attempt = 0
  let stopped = false

  const stop = () => {
    stopped = true
    clearTimeout(retry)
    source.close()
  }

  const scheduleReopen = () => {
    if (stopped) return
    retry = setTimeout(
      () => {
        if (stopped) return
        open()
        handlers.onReopen()
      },
      Math.min(MAX_RETRY_MS, 1000 * 2 ** attempt++),
    )
  }

  // A closed EventSource after a non-200 response may mean the job was deleted
  // rather than the backend restarting. Probe once so a 404 stops the retries.
  const probeJob = () => {
    void api.job(jobId).then(
      () => scheduleReopen(),
      (error: unknown) => {
        if (stopped) return
        if (error instanceof ApiError && isGoneStatus(error.status)) {
          stop()
          handlers.onGone?.(error)
          return
        }
        scheduleReopen()
      },
    )
  }

  const open = () => {
    source = new EventSource(api.eventsUrl(jobId, lastSeq))
    const onMessage = (message: MessageEvent<string>) => {
      const event = parseJobEvent(message.data)
      if (!event) return
      lastSeq = Math.max(lastSeq, event.seq)
      handlers.onEvent(event)
      if (event.type !== 'node') stop()
    }
    for (const type of EVENT_TYPES) source.addEventListener(type, onMessage)
    source.onopen = () => {
      attempt = 0
      handlers.onConnectionChange(true)
    }
    source.onerror = () => {
      handlers.onConnectionChange(false)
      if (stopped || source.readyState !== EventSource.CLOSED) return
      probeJob()
    }
  }

  open()
  return stop
}
