import type { RankedKol } from '@/api/types'
import { flagLabel, isWarningFlag, scoreRows } from './kol'

function row(F_conf: number): RankedKol {
  return {
    scores: { F: 0.5, E: 0.25, T: 1, A_eff: 0.75, F_conf, E_conf: 0.5, T_conf: 0.7, A_conf: 0.6 },
  } as RankedKol
}

describe('scoreRows', () => {
  it('names each score for a marketer, in display order', () => {
    expect(scoreRows(row(0.9)).map((r) => r.label)).toEqual([
      'ตรงกับแบรนด์',
      'การมีส่วนร่วม',
      'ขนาดผู้ติดตามเหมาะสม',
      'กลุ่มผู้ชม (ประมาณการ)',
    ])
  })

  it('marks limited data only below a confidence of 0.5', () => {
    const [low, atThreshold] = scoreRows(row(0.49))
    expect(low.limitedData).toBe(true)
    expect(atThreshold.limitedData).toBe(false)
  })
})

describe('flags', () => {
  it.each([
    'mentions_brand',
    'mentions_competitor',
    'competitor_sponsored',
    'low_reach',
    'activity_unknown',
    'search_stats_only',
    'risky_content',
  ])('gives %s a Thai label', (flag) => {
    expect(flagLabel(flag)).toMatch(/[฀-๿]/)
  })

  it('treats only a brand mention as neutral', () => {
    expect(isWarningFlag('mentions_brand')).toBe(false)
    expect(isWarningFlag('risky_content')).toBe(true)
  })
})
