const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 })

/** 1250000 -> "1.3M", 72000 -> "72K", 950 -> "950". */
export function formatCompact(value: number): string {
  return compact.format(value)
}

/** A 0..1 fraction as a percent with one decimal: 0.091 -> "9.1%". */
export function formatPercent(fraction: number | null): string {
  return fraction === null ? 'n/a' : `${(fraction * 100).toFixed(1)}%`
}

/** A 0..1 score with two decimals: 0.5875 -> "0.59". */
export function formatScore(score: number): string {
  return score.toFixed(2)
}

/** 420 -> "420 ms", 1530 -> "1.5 s", 95000 -> "1m 35s". Rounds before picking the unit. */
export function formatDuration(ms: number): string {
  const whole = Math.round(ms)
  if (whole < 1000) return `${whole} ms`
  const tenths = Math.round(ms / 100) / 10
  if (tenths < 60) return `${tenths.toFixed(1)} s`
  const seconds = Math.round(ms / 1000)
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
}

const dateTime = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' })

export function formatDateTime(iso: string): string {
  return dateTime.format(new Date(iso))
}

/** Two-letter fallback avatar from a handle: "BankZaGaming" -> "BA". */
export function initials(handle: string): string {
  return (
    handle
      .replace(/[^\p{L}\p{N}]/gu, '')
      .slice(0, 2)
      .toUpperCase() || '?'
  )
}
