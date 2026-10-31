import { formatCompact, formatDateTime, hostOf, initials, scoreOutOf100 } from './format'

describe('number formatting', () => {
  it.each([
    [950, '950'],
    [23_000, '23K'],
    [72_000, '72K'],
    [1_250_000, '1.3M'],
    [1_200_000, '1.2M'],
    [0, '0'],
  ])('formats %d followers as %s', (value, expected) => {
    expect(formatCompact(value)).toBe(expected)
  })

  it('puts a score on a 0 to 100 scale', () => {
    expect(scoreOutOf100(0.5875)).toBe(59)
    expect(scoreOutOf100(1)).toBe(100)
    expect(scoreOutOf100(0)).toBe(0)
  })

  it('writes dates the Thai way', () => {
    expect(formatDateTime('2026-10-31T05:00:00Z')).toContain('ต.ค. 2569')
  })

  it('names a site by its host', () => {
    expect(hostOf('https://www.example-brand.co.th/shop')).toBe('example-brand.co.th')
    expect(hostOf('not a url')).toBe('not a url')
  })

  it('builds avatar initials from a handle', () => {
    expect(initials('BankZaGaming')).toBe('BA')
    expect(initials('_x.y')).toBe('XY')
    expect(initials('__')).toBe('?')
  })
})
