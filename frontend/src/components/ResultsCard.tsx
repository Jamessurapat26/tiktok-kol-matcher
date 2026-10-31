import { Download, ExternalLink } from 'lucide-react'
import { useState } from 'react'
import type { KolProfile, RankedKol } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { csvFilename, downloadCsv, toCsv } from '@/lib/csv'
import { formatCompact, initials, scoreOutOf100 } from '@/lib/format'
import { adVideoCount, flagLabel, formatLabel, isWarningFlag, scoreRows } from '@/lib/kol'
import { ScoreBar } from './ScoreBar'

interface Props {
  websiteUrl: string | null
  results: RankedKol[]
  /** Still running: show placeholders instead of an empty list. */
  pending: boolean
  failed: boolean
}

export function ResultsCard({ websiteUrl, results, pending, failed }: Props) {
  const sorted = [...results].sort((a, b) => b.final - a.final)
  return (
    <Card>
      <CardHeader>
        <CardTitle>KOL ที่แนะนำ</CardTitle>
        <CardDescription aria-live="polite">
          {pending
            ? 'กำลังจัดอันดับ KOL…'
            : failed
              ? 'การค้นหาไม่สำเร็จ'
              : `${sorted.length} คน เรียงจากเหมาะสมที่สุด`}
        </CardDescription>
        <CardAction>
          <Button
            variant="outline"
            size="lg"
            disabled={sorted.length === 0}
            onClick={() => downloadCsv(csvFilename(websiteUrl), toCsv(sorted))}
          >
            <Download aria-hidden />
            ดาวน์โหลดรายชื่อ (CSV)
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent>
        {pending ? (
          <div className="flex flex-col gap-3">
            <span className="sr-only">กำลังโหลดผลลัพธ์</span>
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-20" />
            ))}
          </div>
        ) : sorted.length === 0 ? (
          <p className="text-body-sm text-muted-foreground">
            {failed ? 'ไม่มีผลลัพธ์ เพราะการค้นหาไม่สำเร็จ' : 'ไม่พบ KOL ที่เหมาะสม'}
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

/** Profile picture, falling back to the initials circle when missing or broken. */
function KolAvatar({ kol }: { kol: KolProfile }) {
  const [failed, setFailed] = useState(false)
  if (kol.avatar_url && !failed) {
    return (
      <img
        src={kol.avatar_url}
        alt={kol.handle}
        loading="lazy"
        referrerPolicy="no-referrer"
        onError={() => setFailed(true)}
        className="size-10 shrink-0 rounded-full bg-accent-soft object-cover"
      />
    )
  }
  return (
    <span
      aria-hidden
      data-testid="kol-initials"
      className="flex size-10 shrink-0 items-center justify-center rounded-full bg-accent-soft text-body-sm font-bold text-info"
    >
      {initials(kol.handle)}
    </span>
  )
}

function ContentStyle({ row }: { row: RankedKol }) {
  const { content, kol } = row
  const ads = adVideoCount(kol)
  if (!content && ads === 0) return null

  return (
    <section aria-label="สไตล์คอนเทนต์" className="mt-3 border-t border-border pt-3">
      <h4 className="text-body-xs font-bold text-ink">สไตล์คอนเทนต์</h4>
      {content && content.formats.length > 0 && (
        <ul className="mt-1.5 flex flex-wrap gap-1.5" aria-label="รูปแบบคอนเทนต์">
          {content.formats.map((format) => (
            <li key={format}>
              <Badge variant="outline">{formatLabel(format)}</Badge>
            </li>
          ))}
        </ul>
      )}
      {content?.style && <p className="mt-1.5 text-body-sm text-ink">{content.style}</p>}
      {content?.style_fit != null && (
        <p className="mt-1.5 text-body-xs text-muted-foreground">
          <strong className="font-bold text-ink">
            สไตล์เข้ากับแบรนด์ {scoreOutOf100(content.style_fit)}/100
          </strong>
          {' · ยังไม่นับรวมในความเหมาะสมรวม'}
          {content.style_reason && <span className="block">{content.style_reason}</span>}
        </p>
      )}
      {content && content.evidence_urls.length > 0 && (
        <ul className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1" aria-label="คลิปอ้างอิงสไตล์">
          {content.evidence_urls.map((url, index) => (
            <li key={url}>
              <a
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 rounded-sm text-body-xs font-bold text-muted-foreground underline-offset-4 outline-none hover:text-ink hover:underline focus-visible:ring-3 focus-visible:ring-ring"
              >
                คลิปอ้างอิงสไตล์ {index + 1}
                <ExternalLink aria-hidden className="size-3" />
              </a>
            </li>
          ))}
        </ul>
      )}
      {ads > 0 && (
        <p className="mt-1.5 text-body-xs text-muted-foreground">
          ติดป้ายโฆษณา {ads} จาก {kol.videos.length} คลิปที่ตรวจ
        </p>
      )}
    </section>
  )
}

function KolRow({ row, rank }: { row: RankedKol; rank: number }) {
  const { kol } = row

  return (
    <li className="py-4 first:pt-0 last:pb-0">
      <div className="flex gap-4">
        <span className="w-6 shrink-0 pt-2 text-right text-heading-sm text-muted-foreground">
          <span className="sr-only">อันดับ </span>
          {rank}
        </span>
        <KolAvatar kol={kol} />
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
              <span className="sr-only">(เปิด TikTok ในแท็บใหม่)</span>
            </a>
            {kol.nickname && (
              <span className="truncate text-body-sm text-muted-foreground">{kol.nickname}</span>
            )}
          </div>

          <p className="mt-1 text-body-xs text-muted-foreground">
            ผู้ติดตาม <strong className="font-bold text-ink">{formatCompact(kol.followers)}</strong>
          </p>

          <div className="mt-3 flex items-baseline justify-between gap-3 text-body-sm">
            <span className="font-bold text-ink">ความเหมาะสมรวม</span>
            <span className="font-bold text-ink">{scoreOutOf100(row.final)}/100</span>
          </div>
          <ScoreBar value={row.final} label="ความเหมาะสมรวม" className="mt-1" />

          <dl className="mt-3 grid gap-x-6 gap-y-2 sm:grid-cols-2">
            {scoreRows(row).map((score) => (
              <div key={score.key}>
                <div className="flex items-baseline justify-between gap-2 text-body-xs text-muted-foreground">
                  <dt>{score.label}</dt>
                  <dd className="flex shrink-0 items-baseline gap-2">
                    {score.limitedData && <span className="text-warning">ข้อมูลน้อย</span>}
                    <span className="font-bold text-ink">{scoreOutOf100(score.value)}</span>
                  </dd>
                </div>
                <ScoreBar value={score.value} label={score.label} className="mt-1 h-1.5" />
              </div>
            ))}
          </dl>

          {row.flags.length > 0 && (
            <ul className="mt-3 flex flex-wrap gap-1.5" aria-label="ป้ายเตือน">
              {row.flags.map((flag) => (
                <li key={flag}>
                  <Badge variant={isWarningFlag(flag) ? 'warning' : 'outline'}>{flagLabel(flag)}</Badge>
                </li>
              ))}
            </ul>
          )}

          <p className="mt-2 text-body-sm text-ink">{row.reason}</p>

          <ContentStyle row={row} />

          {row.evidence_urls.length > 0 && (
            <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1" aria-label="วิดีโอตัวอย่าง">
              {row.evidence_urls.slice(0, 3).map((url, index) => (
                <li key={url}>
                  <a
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 rounded-sm text-body-xs font-bold text-muted-foreground underline-offset-4 outline-none hover:text-ink hover:underline focus-visible:ring-3 focus-visible:ring-ring"
                  >
                    วิดีโอตัวอย่าง {index + 1}
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
