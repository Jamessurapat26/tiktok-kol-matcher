import { formatCompact, formatDuration, formatPercent, formatScore, initials } from './format'

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

  it('formats an engagement fraction as a percent', () => {
    expect(formatPercent(0.091)).toBe('9.1%')
    expect(formatPercent(0.0434)).toBe('4.3%')
    expect(formatPercent(0)).toBe('0.0%')
    expect(formatPercent(null)).toBe('n/a')
  })

  it('formats scores with two decimals', () => {
    expect(formatScore(0.5875)).toBe('0.59')
    expect(formatScore(1)).toBe('1.00')
  })

  it('formats durations by magnitude', () => {
    expect(formatDuration(3.4)).toBe('3 ms')
    expect(formatDuration(1530)).toBe('1.5 s')
    expect(formatDuration(95_000)).toBe('1m 35s')
  })

  it('builds avatar initials from a handle', () => {
    expect(initials('BankZaGaming')).toBe('BA')
    expect(initials('_x.y')).toBe('XY')
    expect(initials('__')).toBe('?')
  })
})
