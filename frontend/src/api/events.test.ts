import demoEvents from '@/test/demo-events.json'
import { parseJobEvent } from './events'

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
