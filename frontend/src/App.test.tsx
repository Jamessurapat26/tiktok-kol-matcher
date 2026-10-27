import { render, screen } from '@testing-library/react'
import App from './App'

vi.mock('@/hooks/useJobParam', () => ({
  useJobParam: () => ['missing', vi.fn()],
}))

vi.mock('@/hooks/useJobProgress', async () => {
  const { initialProgress } = await import('@/lib/steps')
  return { useJobProgress: () => ({ progress: initialProgress(), connected: false }) }
})

vi.mock('@/api/queries', () => ({
  useJob: () => ({
    isError: true,
    error: { message: 'Job not found' },
    data: undefined,
  }),
  useHealth: () => ({ data: undefined, isError: false }),
  useJobs: () => ({ isPending: false, isError: false, data: [] }),
  useCreateJob: () => ({ mutate: vi.fn(), isPending: false, error: null, reset: vi.fn() }),
}))

describe('App', () => {
  it('hides the progress card when the job query fails', () => {
    render(<App />)

    expect(screen.getByRole('alert')).toHaveTextContent('Could not load job missing: Job not found')
    expect(screen.queryByText('Progress')).not.toBeInTheDocument()
  })
})
