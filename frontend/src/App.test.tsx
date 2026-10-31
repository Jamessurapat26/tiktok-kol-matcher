import { render, screen } from '@testing-library/react'
import App from './App'

vi.mock('@/hooks/useJobParam', () => ({
  useJobParam: () => ['missing', vi.fn()],
}))

vi.mock('@/hooks/useJobProgress', async () => {
  const { initialProgress } = await import('@/lib/steps')
  return { useJobProgress: () => ({ progress: initialProgress(), connected: false }) }
})

vi.mock('@/api/queries', async () => {
  const { ApiError } = await import('@/api/client')
  return {
    useJob: () => ({
      isError: true,
      error: new ApiError(404, { detail: 'job missing not found' }),
      data: undefined,
    }),
    useHealth: () => ({ data: undefined, isError: false }),
    useJobs: () => ({ isPending: false, isError: false, data: [] }),
    useCreateJob: () => ({ mutate: vi.fn(), isPending: false, error: null, reset: vi.fn() }),
  }
})

describe('App', () => {
  it('hides the progress card when the job query fails', () => {
    render(<App />)

    expect(screen.getByRole('alert')).toHaveTextContent('ไม่พบการค้นหานี้')
    expect(screen.queryByText('ความคืบหน้า')).not.toBeInTheDocument()
  })
})
