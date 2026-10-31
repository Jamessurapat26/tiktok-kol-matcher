import type { ContentFormat, KolProfile, RankedKol } from '@/api/types'

/** Below this confidence a score is marked as limited data. */
const LOW_CONFIDENCE = 0.5

export interface ScoreRow {
  key: 'F' | 'E' | 'T' | 'A'
  label: string
  value: number
  limitedData: boolean
}

export function scoreRows({ scores }: RankedKol): ScoreRow[] {
  const row = (key: ScoreRow['key'], label: string, value: number, confidence: number): ScoreRow => ({
    key,
    label,
    value,
    limitedData: confidence < LOW_CONFIDENCE,
  })
  return [
    row('F', 'ตรงกับแบรนด์', scores.F, scores.F_conf),
    row('E', 'การมีส่วนร่วม', scores.E, scores.E_conf),
    row('T', 'ขนาดผู้ติดตามเหมาะสม', scores.T, scores.T_conf),
    row('A', 'กลุ่มผู้ชม (ประมาณการ)', scores.A_eff, scores.A_conf),
  ]
}

const FORMAT_LABELS: Record<ContentFormat, string> = {
  review: 'รีวิว',
  tutorial: 'สอน/แนะนำวิธี',
  unboxing: 'แกะกล่อง',
  comparison: 'เปรียบเทียบ',
  vlog: 'Vlog',
  comedy: 'ตลก',
  storytelling: 'เล่าเรื่อง',
  challenge: 'ชาเลนจ์',
  lifestyle: 'ไลฟ์สไตล์',
  news: 'ข่าว/อัปเดต',
}

export function formatLabel(format: ContentFormat): string {
  return FORMAT_LABELS[format]
}

/** Recent videos TikTok labels as ads. Unlabelled videos can still be sponsored. */
export function adVideoCount(kol: KolProfile): number {
  return kol.videos.filter((video) => video.is_ad).length
}

const FLAG_LABELS: Record<string, string> = {
  mentions_brand: 'เคยพูดถึงแบรนด์นี้',
  mentions_competitor: 'เคยพูดถึงคู่แข่ง',
  competitor_sponsored: 'เพิ่งรับงานคู่แข่ง',
  low_reach: 'ยอดวิวต่ำเมื่อเทียบกับผู้ติดตาม',
  activity_unknown: 'ไม่ทราบวันที่โพสต์ล่าสุด',
  search_stats_only: 'ข้อมูลสถิติจำกัด',
  risky_content: 'เนื้อหาเสี่ยง เช่น การพนัน',
  unreviewed: 'AI ยังไม่ได้ตรวจบัญชีนี้',
}

export function flagLabel(flag: string): string {
  return FLAG_LABELS[flag] ?? flag
}

/** Mentioning the brand is neutral news; every other flag is something to check. */
export function isWarningFlag(flag: string): boolean {
  return flag !== 'mentions_brand'
}
