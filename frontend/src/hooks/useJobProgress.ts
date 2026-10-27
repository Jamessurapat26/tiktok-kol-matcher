import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { followJobEvents } from '@/api/events'
import { queryKeys } from '@/api/queries'
import type { JobEvent } from '@/api/types'
import { applyEvent, initialProgress, type Progress } from '@/lib/steps'

// The runner saves brand_profile right after analyze_brand's finished event and
// before the next node starts, so generate_queries starting is the first event
// that guarantees the profile is readable.
function refreshesJob(event: JobEvent): boolean {
  if (event.type !== 'node') return true
  return (
    (event.node === 'analyze_brand' && event.status !== 'started') ||
    (event.node === 'generate_queries' && event.status === 'started')
  )
}

interface State {
  jobId: string | null
  progress: Progress
  connected: boolean
}

/** Live step progress for a job, from its SSE stream. */
export function useJobProgress(jobId: string | null): { progress: Progress; connected: boolean } {
  const client = useQueryClient()
  const [state, setState] = useState<State>({ jobId, progress: initialProgress(), connected: false })

  useEffect(() => {
    if (!jobId) return
    const forJob = (s: State): State =>
      s.jobId === jobId ? s : { jobId, progress: initialProgress(), connected: false }
    return followJobEvents(jobId, {
      onEvent: (event) => {
        setState((s) => {
          const current = forJob(s)
          return { ...current, progress: applyEvent(current.progress, event) }
        })
        if (refreshesJob(event)) void client.invalidateQueries({ queryKey: queryKeys.job(jobId) })
        if (event.type !== 'node') void client.invalidateQueries({ queryKey: queryKeys.jobs, exact: true })
      },
      onConnectionChange: (connected) => setState((s) => ({ ...forJob(s), connected })),
      onReopen: () => void client.invalidateQueries({ queryKey: queryKeys.job(jobId) }),
    })
  }, [jobId, client])

  // Until the new job's first event arrives, show a fresh stepper, not the old job's.
  return state.jobId === jobId ? state : { progress: initialProgress(), connected: false }
}
