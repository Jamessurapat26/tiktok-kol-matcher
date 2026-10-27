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
}

/**
 * Follow a job's events. EventSource reconnects on its own and sends
 * Last-Event-ID, so the server resumes after the last seq it delivered.
 * The stream is closed on the terminal event; otherwise EventSource would
 * reconnect to a finished job forever.
 */
export function followJobEvents(jobId: string, handlers: EventStreamHandlers): () => void {
  const source = new EventSource(api.eventsUrl(jobId))
  const onMessage = (message: MessageEvent<string>) => {
    const event = parseJobEvent(message.data)
    if (!event) return
    handlers.onEvent(event)
    if (event.type !== 'node') source.close()
  }
  for (const type of EVENT_TYPES) source.addEventListener(type, onMessage)
  source.onopen = () => handlers.onConnectionChange(true)
  source.onerror = () => handlers.onConnectionChange(false)
  return () => source.close()
}
