import type { RankedKol } from '@/api/types'

const COLUMNS: [string, (row: RankedKol) => string | number | null][] = [
  ['handle', (r) => r.kol.handle],
  ['nickname', (r) => r.kol.nickname],
  ['url', (r) => r.kol.url],
  ['followers', (r) => r.kol.followers],
  ['engagement_rate', (r) => r.kol.median_engagement_rate],
  ['total_score', (r) => r.total_score],
  ['relevance', (r) => r.scores.relevance],
  ['engagement', (r) => r.scores.engagement],
  ['audience_fit', (r) => r.scores.audience_fit],
  ['tier_fit', (r) => r.scores.tier_fit],
  ['audience_confidence', (r) => r.scores.audience_confidence],
  ['reason', (r) => r.reason],
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
  const header = ['rank', ...COLUMNS.map(([name]) => name)].join(',')
  const lines = rows.map((row, i) => [i + 1, ...COLUMNS.map(([, get]) => cell(get(row)))].join(','))
  return [header, ...lines].join('\r\n') + '\r\n'
}

export function downloadCsv(filename: string, csv: string): void {
  // The BOM makes Excel read the file as UTF-8, so Thai nicknames survive.
  const url = URL.createObjectURL(new Blob(['﻿', csv], { type: 'text/csv;charset=utf-8' }))
  const link = Object.assign(document.createElement('a'), { href: url, download: filename })
  link.click()
  URL.revokeObjectURL(url)
}
