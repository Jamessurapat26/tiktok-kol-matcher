const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 })

/** 1250000 -> "1.3M", 72000 -> "72K", 950 -> "950". */
export function formatCompact(value: number): string {
  return compact.format(value)
}

/** A 0..1 score on a 0..100 scale: 0.5875 -> 59. */
export function scoreOutOf100(score: number): number {
  return Math.round(score * 100)
}

const dateTime = new Intl.DateTimeFormat('th-TH', { dateStyle: 'medium', timeStyle: 'short' })

export function formatDateTime(iso: string): string {
  return dateTime.format(new Date(iso))
}

/** "https://www.example.co.th/shop" -> "example.co.th". Anything unparsable comes back as is. */
export function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return url
  }
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
