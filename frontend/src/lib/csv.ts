import type { RankedKol } from '@/api/types'
import { hostOf, scoreOutOf100 } from './format'
import { flagLabel, type ScoreRow, scoreRows } from './kol'

const score = (key: ScoreRow['key']) => (row: RankedKol) =>
  scoreOutOf100(scoreRows(row).find((r) => r.key === key)?.value ?? 0)

const COLUMNS: [string, (row: RankedKol) => string | number | null][] = [
  ['ชื่อบัญชี', (r) => r.kol.handle],
  ['ชื่อที่แสดง', (r) => r.kol.nickname],
  ['ลิงก์โปรไฟล์', (r) => r.profile_url],
  ['ผู้ติดตาม', (r) => r.kol.followers],
  ['ความเหมาะสมรวม', (r) => scoreOutOf100(r.final)],
  ['ตรงกับแบรนด์', score('F')],
  ['การมีส่วนร่วม', score('E')],
  ['ขนาดผู้ติดตามเหมาะสม', score('T')],
  ['กลุ่มผู้ชม (ประมาณการ)', score('A')],
  [
    'คะแนนที่ข้อมูลน้อย',
    (r) =>
      scoreRows(r)
        .filter((row) => row.limitedData)
        .map((row) => row.label)
        .join('; '),
  ],
  ['ป้ายเตือน', (r) => r.flags.map(flagLabel).join('; ')],
  ['เหตุผล', (r) => r.reason],
  ['วิดีโอตัวอย่าง', (r) => r.evidence_urls.join(' | ')],
]

function cell(value: string | number | null): string {
  if (value === null) return ''
  let text = String(value)
  // Creator text is untrusted: a leading = + - @ would run as a spreadsheet formula.
  if (typeof value === 'string' && /^[=+\-@\t\r]/.test(text)) text = `'${text}`
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
}

/** Ranked KOLs as CSV, in the given order, with a 1-based rank column. */
export function toCsv(rows: RankedKol[]): string {
  const header = ['อันดับ', ...COLUMNS.map(([name]) => name)].join(',')
  const lines = rows.map((row, i) => [i + 1, ...COLUMNS.map(([, get]) => cell(get(row)))].join(','))
  return [header, ...lines].join('\r\n') + '\r\n'
}

/** "KOL-example-brand.co.th-2026-10-31.csv", dated in the viewer's time zone. */
export function csvFilename(websiteUrl: string | null, now = new Date()): string {
  const date = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
  ].join('-')
  return ['KOL', ...(websiteUrl ? [hostOf(websiteUrl)] : []), `${date}.csv`].join('-')
}

export function downloadCsv(filename: string, csv: string): void {
  // The BOM makes Excel read the file as UTF-8, so Thai nicknames survive.
  const url = URL.createObjectURL(new Blob(['﻿', csv], { type: 'text/csv;charset=utf-8' }))
  const link = Object.assign(document.createElement('a'), { href: url, download: filename })
  link.click()
  URL.revokeObjectURL(url)
}
