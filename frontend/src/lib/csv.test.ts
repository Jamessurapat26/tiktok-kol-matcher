import type { RankedKol } from '@/api/types'
import { toCsv } from './csv'

function ranked(
  handle: string,
  total: number,
  overrides: Partial<RankedKol['kol']> = {},
  reason = 'Good fit.',
) {
  return {
    kol: {
      handle,
      url: `https://www.tiktok.com/@${handle}`,
      nickname: null,
      followers: 1000,
      bio: null,
      language: 'th',
      region: 'TH',
      median_engagement_rate: 0.05,
      posts_per_week: 2,
      content_summary: null,
      source: 'seed',
      ...overrides,
    },
    scores: {
      relevance: 0.5,
      engagement: 0.25,
      audience_fit: 1,
      tier_fit: 0.75,
      audience_confidence: 'high',
    },
    total_score: total,
    reason,
  } satisfies RankedKol
}

const parseRows = (csv: string) => csv.trimEnd().split('\r\n')

describe('toCsv', () => {
  it('writes a header and one ranked row per KOL in order', () => {
    const rows = parseRows(toCsv([ranked('alpha', 0.9), ranked('beta', 0.7)]))
    expect(rows[0]).toBe(
      'rank,handle,nickname,url,followers,engagement_rate,total_score,relevance,engagement,audience_fit,tier_fit,audience_confidence,reason',
    )
    expect(rows[1]).toBe(
      '1,alpha,,https://www.tiktok.com/@alpha,1000,0.05,0.9,0.5,0.25,1,0.75,high,Good fit.',
    )
    expect(rows[2].startsWith('2,beta,')).toBe(true)
    expect(rows).toHaveLength(3)
  })

  it('quotes commas, quotes, and newlines, and keeps Thai text', () => {
    const csv = toCsv([ranked('alpha', 0.9, { nickname: 'แบงค์ "ซ่า"' }, 'Tier: mega,\naudience fit: high.')])
    expect(csv).toContain(',"แบงค์ ""ซ่า""",')
    expect(csv).toContain(',"Tier: mega,\naudience fit: high."\r\n')
  })

  it('neutralizes text that a spreadsheet would run as a formula', () => {
    const csv = toCsv([ranked('alpha', 0.9, { nickname: '=HYPERLINK("x")' }, '+1 great')])
    expect(csv).toContain(`,"'=HYPERLINK(""x"")",`)
    expect(csv).toContain(",'+1 great\r\n")
  })

  it('leaves an empty cell for a missing engagement rate', () => {
    const rows = parseRows(toCsv([ranked('alpha', 0.9, { median_engagement_rate: null })]))
    expect(rows[1]).toContain(',1000,,0.9,')
  })
})
