import body422 from '@/test/error-422.json'
import { ApiError } from './client'

describe('ApiError', () => {
  it('maps a real 422 body to per-field messages', () => {
    const error = new ApiError(422, body422)
    expect(error.fields).toEqual({
      website_url: 'website_url must not point at a private or local address',
      facebook_url: 'facebook_url must be a facebook.com or fb.com page',
      top_k: 'Input should be greater than or equal to 1',
    })
  })

  it('uses the detail string of a 429 as its message', () => {
    const error = new ApiError(429, { detail: '20 jobs are already queued or running' })
    expect(error.message).toBe('20 jobs are already queued or running')
    expect(error.fields).toEqual({})
  })

  it('falls back to the status for a body without detail', () => {
    expect(new ApiError(502, null).message).toBe('Request failed (502)')
  })
})
