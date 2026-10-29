import { Download, ExternalLink } from 'lucide-react'
import type { RankedKol } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { downloadCsv, toCsv } from '@/lib/csv'
import { formatCompact, formatScore, initials } from '@/lib/format'
import { ScoreBar } from './ScoreBar'

interface Props {
  jobId: string
  results: RankedKol[]
  /** Still running: show placeholders instead of an empty list. */
  pending: boolean
  failed: boolean
}

export function ResultsCard({ jobId, results, pending, failed }: Props) {
  const sorted = [...results].sort((a, b) => b.final - a.final)
  return (
    <Card>
      <CardHeader>
        <CardTitle>Ranked KOLs</CardTitle>
        <CardDescription aria-live="polite">
          {pending
            ? 'Ranking creators…'
            : failed
              ? 'Run failed'
              : `${sorted.length} creators, best match first`}
        </CardDescription>
        <CardAction>
          <Button
            variant="outline"
            size="lg"
            disabled={sorted.length === 0}
            onClick={() => downloadCsv(`kols-${jobId}.csv`, toCsv(sorted))}
          >
            <Download aria-hidden />
            Export CSV
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent>
        {pending ? (
          <div className="flex flex-col gap-3">
            <span className="sr-only">Loading results</span>
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-20" />
            ))}
          </div>
        ) : sorted.length === 0 ? (
          <p className="text-body-sm text-muted-foreground">
            {failed ? 'No results: the run failed.' : 'No creators matched.'}
          </p>
        ) : (
          <ol className="flex flex-col divide-y divide-border">
            {sorted.map((row, i) => (
              <KolRow key={row.kol.handle} row={row} rank={i + 1} />
            ))}
          </ol>
        )}
      </CardContent>
    </Card>
  )
}

function KolRow({ row, rank }: { row: RankedKol; rank: number }) {
  const { kol, scores } = row
  const scoreRows = [
    { key: 'F', label: 'Fit (F)', value: scores.F, confidence: scores.F_conf },
    { key: 'E', label: 'Engagement (E)', value: scores.E, confidence: scores.E_conf },
    { key: 'T', label: 'Tier fit (T)', value: scores.T, confidence: scores.T_conf },
    { key: 'A', label: 'Audience (A_eff)', value: scores.A_eff, confidence: scores.A_conf },
  ] as const

  return (
    <li className="py-4 first:pt-0 last:pb-0">
      <div className="flex gap-4">
        <span className="w-6 shrink-0 pt-2 text-right text-heading-sm text-muted-foreground">
          <span className="sr-only">Rank </span>
          {rank}
        </span>
        <span
          aria-hidden
          className="flex size-10 shrink-0 items-center justify-center rounded-full bg-accent-soft text-body-sm font-bold text-info"
        >
          {initials(kol.handle)}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <a
              href={row.profile_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 rounded-sm text-body-md font-bold text-ink underline-offset-4 outline-none hover:underline focus-visible:ring-3 focus-visible:ring-ring"
            >
              @{kol.handle}
              <ExternalLink aria-hidden className="size-3.5 text-muted-foreground" />
              <span className="sr-only">(opens TikTok in a new tab)</span>
            </a>
            {kol.nickname && (
              <span className="truncate text-body-sm text-muted-foreground">{kol.nickname}</span>
            )}
          </div>

          <div className="mt-1 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-body-xs text-muted-foreground">
            <span>
              <strong className="font-bold text-ink">{formatCompact(kol.followers)}</strong> followers
            </span>
            <span>
              Final <strong className="font-bold text-ink">{formatScore(row.final)}</strong>
            </span>
            <span>
              Quality <strong className="font-bold text-ink">{formatScore(row.quality)}</strong>
            </span>
          </div>

          <div className="mt-3 flex items-center gap-3">
            <ScoreBar value={row.final} label="Final score" className="flex-1" />
            <span className="w-10 text-right text-body-sm font-bold text-ink">{formatScore(row.final)}</span>
          </div>

          <dl className="mt-3 grid gap-x-6 gap-y-2 sm:grid-cols-2">
            {scoreRows.map((score) => (
              <div key={score.key} className="grid grid-cols-[7.5rem_1fr_3.5rem] items-center gap-2">
                <dt className="text-body-xs text-muted-foreground">{score.label}</dt>
                <dd>
                  <ScoreBar value={score.value} label={score.label} className="h-1.5" />
                </dd>
                <dd className="text-right text-body-xs text-muted-foreground">
                  <span className="font-bold text-ink">{formatScore(score.value)}</span>
                  <span className="block">conf {formatScore(score.confidence)}</span>
                </dd>
              </div>
            ))}
          </dl>

          {row.flags.length > 0 && (
            <ul className="mt-3 flex flex-wrap gap-1.5" aria-label="Flags">
              {row.flags.map((flag) => (
                <li key={flag}>
                  <Badge variant="outline">{flag}</Badge>
                </li>
              ))}
            </ul>
          )}

          <p className="mt-2 text-body-sm text-ink">{row.reason}</p>

          {row.evidence_urls.length > 0 && (
            <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1" aria-label="Evidence videos">
              {row.evidence_urls.slice(0, 3).map((url, index) => (
                <li key={url}>
                  <a
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 rounded-sm text-body-xs font-bold text-muted-foreground underline-offset-4 outline-none hover:text-ink hover:underline focus-visible:ring-3 focus-visible:ring-ring"
                  >
                    Evidence {index + 1}
                    <ExternalLink aria-hidden className="size-3" />
                  </a>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </li>
  )
}
