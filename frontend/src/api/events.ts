import { api } from './client'
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
}

const MAX_RETRY_MS = 10_000

/**
 * Follow a job's events. EventSource reconnects on its own and sends
 * Last-Event-ID, so the server resumes after the last seq it delivered.
 * A reconnect that gets a non-200 (a 502 while the backend restarts) closes
 * the EventSource for good, so that case reopens one after a backoff with
 * `?after=<last seq>`. The stream is closed on the terminal event; otherwise
 * it would reconnect to a finished job forever.
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
      retry = setTimeout(
        () => {
          open()
          handlers.onReopen()
        },
        Math.min(MAX_RETRY_MS, 1000 * 2 ** attempt++),
      )
    }
  }

  open()
  return stop
}
