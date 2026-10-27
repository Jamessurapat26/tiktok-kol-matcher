import { ChevronDown, Download, ExternalLink } from 'lucide-react'
import { useState } from 'react'
import type { AudienceConfidence, RankedKol } from '@/api/types'
import { Button } from '@/components/ui/button'
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { downloadCsv, toCsv } from '@/lib/csv'
import { formatCompact, formatPercent, formatScore, initials } from '@/lib/format'
import { cn } from '@/lib/utils'
import { ScoreBar } from './ScoreBar'

const CONFIDENCE: Record<AudienceConfidence, number> = { low: 1 / 3, medium: 2 / 3, high: 1 }

interface Props {
  jobId: string
  results: RankedKol[]
  /** Still running: show placeholders instead of an empty list. */
  pending: boolean
  failed: boolean
}

export function ResultsCard({ jobId, results, pending, failed }: Props) {
  const sorted = [...results].sort((a, b) => b.total_score - a.total_score)
  return (
    <Card>
      <CardHeader>
        <CardTitle>Ranked KOLs</CardTitle>
        <CardDescription>
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
          <div className="flex flex-col gap-3" aria-label="Loading results">
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
  const [open, setOpen] = useState(false)
  const { kol, scores } = row
  const detailId = `kol-${rank}-breakdown`

  return (
    <li className="py-4 first:pt-0 last:pb-0">
      <div className="flex gap-4">
        <span
          className="w-6 shrink-0 pt-2 text-right text-heading-sm text-subtle"
          aria-label={`Rank ${rank}`}
        >
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
              href={kol.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 rounded-sm text-body-md font-bold text-ink underline-offset-4 outline-none hover:underline focus-visible:ring-3 focus-visible:ring-ring/50"
            >
              @{kol.handle}
              <ExternalLink aria-hidden className="size-3.5 text-muted-foreground" />
              <span className="sr-only">(opens TikTok in a new tab)</span>
            </a>
            {kol.nickname && (
              <span className="truncate text-body-sm text-muted-foreground">{kol.nickname}</span>
            )}
          </div>
          <div className="mt-1 flex flex-wrap gap-x-4 text-body-xs text-muted-foreground">
            <span>
              <strong className="font-bold text-ink">{formatCompact(kol.followers)}</strong> followers
            </span>
            <span>
              <strong className="font-bold text-ink">{formatPercent(kol.median_engagement_rate)}</strong>{' '}
              engagement
            </span>
          </div>
          <div className="mt-3 flex items-center gap-3">
            <ScoreBar value={row.total_score} label="Total score" className="flex-1" />
            <span className="w-10 text-right text-body-sm font-bold text-ink">
              {formatScore(row.total_score)}
            </span>
          </div>
          <p className="mt-2 text-body-sm text-ink">{row.reason}</p>
          <button
            type="button"
            aria-expanded={open}
            aria-controls={detailId}
            onClick={() => setOpen((o) => !o)}
            className="mt-2 inline-flex items-center gap-1 rounded-sm text-body-xs font-bold text-muted-foreground outline-none hover:text-ink focus-visible:ring-3 focus-visible:ring-ring/50"
          >
            Score breakdown
            <ChevronDown aria-hidden className={cn('size-4 transition-transform', open && 'rotate-180')} />
          </button>
          {open && (
            <dl
              id={detailId}
              className="mt-3 grid grid-cols-[8rem_1fr_3rem] items-center gap-x-3 gap-y-2 text-body-xs"
            >
              <Breakdown label="Relevance" value={scores.relevance} />
              <Breakdown label="Engagement" value={scores.engagement} />
              <Breakdown label="Audience fit" value={scores.audience_fit} />
              <Breakdown label="Tier fit" value={scores.tier_fit} />
              <Breakdown
                label="Audience confidence"
                value={CONFIDENCE[scores.audience_confidence]}
                text={scores.audience_confidence}
              />
            </dl>
          )}
        </div>
      </div>
    </li>
  )
}

function Breakdown({ label, value, text }: { label: string; value: number; text?: string }) {
  return (
    <>
      <dt className="text-muted-foreground">{label}</dt>
      <dd>
        <ScoreBar value={value} label={label} className="h-1.5" />
      </dd>
      <dd className="text-right font-bold text-ink">{text ?? formatScore(value)}</dd>
    </>
  )
}
