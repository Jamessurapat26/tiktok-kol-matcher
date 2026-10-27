import type { ErrorBody, Health, Job, JobCreate, JobCreated, ValidationIssue } from './types'

export class ApiError extends Error {
  readonly status: number
  /** Field name -> message, from a 422 body's `loc` and `msg`. */
  readonly fields: Record<string, string>

  constructor(status: number, body: unknown) {
    const detail = isErrorBody(body) ? body.detail : undefined
    const issues = Array.isArray(detail) ? detail : []
    const message =
      typeof detail === 'string' ? detail : issues.map((issue) => cleanMessage(issue.msg)).join('; ')
    super(message || `Request failed (${status})`)
    this.status = status
    this.fields = Object.fromEntries(issues.map((issue) => [fieldOf(issue), cleanMessage(issue.msg)]))
  }
}

function isErrorBody(body: unknown): body is ErrorBody {
  return typeof body === 'object' && body !== null && 'detail' in body
}

function fieldOf(issue: ValidationIssue): string {
  return String(issue.loc.at(-1) ?? '')
}

// Pydantic prefixes custom validator messages with "Value error, ".
function cleanMessage(msg: string): string {
  return msg.replace(/^Value error, /, '')
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'content-type': 'application/json', ...init?.headers },
  })
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) throw new ApiError(response.status, body)
  return body as T
}

export const api = {
  health: () => request<Health>('/api/health'),
  jobs: (limit = 20) => request<Job[]>(`/api/jobs?limit=${limit}`),
  job: (id: string) => request<Job>(`/api/jobs/${encodeURIComponent(id)}`),
  createJob: (body: JobCreate) =>
    request<JobCreated>('/api/jobs', { method: 'POST', body: JSON.stringify(body) }),
  eventsUrl: (id: string) => `/api/jobs/${encodeURIComponent(id)}/events`,
}
