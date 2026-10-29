import type { RankedKol } from '@/api/types'
import { toCsv } from './csv'

function ranked(
  handle: string,
  final: number,
  overrides: Partial<RankedKol['kol']> = {},
  reason = 'Good fit.',
  scoreOverrides: Partial<RankedKol['scores']> = {},
): RankedKol {
  return {
    kol: {
      handle,
      url: `https://www.tiktok.com/@${handle}`,
      nickname: null,
      followers: 1000,
      bio: null,
      language: 'th',
      region: 'TH',
      posts_per_week: 2,
      content_summary: null,
      videos: [],
      stats_source: 'search',
      source: 'seed',
      ...overrides,
    },
    scores: {
      F: 0.5,
      E: 0.25,
      T: 1,
      A_eff: 0.75,
      F_conf: 0.9,
      E_conf: 0.8,
      T_conf: 0.7,
      A_conf: 0.6,
      sim: 0.4,
      on_topic: 0.3,
      judge: 0.5,
      er_star: 0.1,
      reach: 0.2,
      judged: true,
      ...scoreOverrides,
    },
    quality: 0.5,
    final,
    flags: ['mentions_brand'],
    reason,
    evidence_urls: [`https://www.tiktok.com/@${handle}/video/1`],
    profile_url: `https://www.tiktok.com/@${handle}`,
  }
}

const parseRows = (csv: string) => csv.trimEnd().split('\r\n')

describe('toCsv', () => {
  it('writes a header and one ranked row per KOL in order', () => {
    const rows = parseRows(toCsv([ranked('alpha', 0.9), ranked('beta', 0.7)]))
    expect(rows[0]).toBe(
      'rank,handle,nickname,profile_url,followers,final,F,E,T,A_eff,F_conf,E_conf,T_conf,A_conf,flags,reason,evidence_urls',
    )
    expect(rows[1]).toBe(
      '1,alpha,,https://www.tiktok.com/@alpha,1000,0.9,0.5,0.25,1,0.75,0.9,0.8,0.7,0.6,mentions_brand,Good fit.,https://www.tiktok.com/@alpha/video/1',
    )
    expect(rows[2].startsWith('2,beta,')).toBe(true)
    expect(rows).toHaveLength(3)
  })

  it('joins flags and evidence links into one cell each', () => {
    const row = {
      ...ranked('alpha', 0.9),
      flags: ['mentions_brand', 'low_reach'],
      evidence_urls: ['https://www.tiktok.com/@alpha/video/1', 'https://www.tiktok.com/@alpha/video/2'],
    }
    const rows = parseRows(toCsv([row]))
    expect(rows[1]).toContain(
      ',mentions_brand; low_reach,Good fit.,https://www.tiktok.com/@alpha/video/1 | https://www.tiktok.com/@alpha/video/2',
    )
  })

  it('quotes commas, quotes, and newlines, and keeps Thai text', () => {
    const csv = toCsv([
      ranked('alpha', 0.9, { nickname: 'แบงค์ "ซ่า"' }, 'Tier: micro,\naudience fit: high.'),
    ])
    expect(csv).toContain(',"แบงค์ ""ซ่า""",')
    expect(csv).toContain(',"Tier: micro,\naudience fit: high.",https://www.tiktok.com/@alpha/video/1')
  })

  it('neutralizes text that a spreadsheet would run as a formula', () => {
    const csv = toCsv([ranked('alpha', 0.9, { nickname: '=HYPERLINK("x")' }, '+1 great')])
    expect(csv).toContain(`,"'=HYPERLINK(""x"")",`)
    expect(csv).toContain(",'+1 great,https://www.tiktok.com/@alpha/video/1")
  })

  it('leaves empty cells for no flags and no evidence', () => {
    const row = { ...ranked('alpha', 0.9), flags: [], evidence_urls: [] }
    const rows = parseRows(toCsv([row]))
    expect(rows[1].endsWith(',0.6,,Good fit.,')).toBe(true)
  })
})
