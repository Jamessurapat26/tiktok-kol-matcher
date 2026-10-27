import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './client'

export const queryKeys = {
  health: ['health'] as const,
  jobs: ['jobs'] as const,
  job: (id: string) => ['jobs', id] as const,
}

export function useHealth() {
  return useQuery({ queryKey: queryKeys.health, queryFn: api.health, staleTime: 60_000 })
}

export function useJobs(enabled: boolean) {
  return useQuery({ queryKey: queryKeys.jobs, queryFn: () => api.jobs(), enabled })
}

export function useJob(id: string | null) {
  return useQuery({
    queryKey: queryKeys.job(id ?? ''),
    queryFn: () => api.job(id ?? ''),
    enabled: id !== null,
  })
}

export function useCreateJob() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: api.createJob,
    onSuccess: () => client.invalidateQueries({ queryKey: queryKeys.jobs, exact: true }),
  })
}
