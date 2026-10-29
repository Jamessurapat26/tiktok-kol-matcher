import { Check, ChevronDown, Circle, Loader2, X } from 'lucide-react'
import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { formatDateTime, formatDuration } from '@/lib/format'
import { type Progress, STAGE_LABELS, type Step, type StepStatus, stepDuration } from '@/lib/steps'
import { cn } from '@/lib/utils'

export function ProgressCard({ progress, connected }: { progress: Progress; connected: boolean }) {
  const { outcome } = progress
  const running = progress.steps.find((s) => s.status === 'running')
  const status = outcome
    ? outcome.type === 'done'
      ? `Done: ${outcome.summary}`
      : `Failed: ${outcome.summary}`
    : running
      ? `Running: ${STAGE_LABELS[running.stage]}`
      : connected
        ? 'Waiting for the pipeline to start'
        : 'Connecting…'

  return (
    <Card>
      <CardHeader>
        <CardTitle>Progress</CardTitle>
        <CardDescription aria-live="polite" className={cn(outcome?.type === 'failed' && 'text-error')}>
          {status}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ol className="flex flex-col">
          {progress.steps.map((step, i) => (
            <StepRow key={step.stage} step={step} last={i === progress.steps.length - 1} />
          ))}
        </ol>
      </CardContent>
    </Card>
  )
}

function StepRow({ step, last }: { step: Step; last: boolean }) {
  const [open, setOpen] = useState(false)
  const duration = stepDuration(step)
  const detailId = `step-${step.stage}-detail`
  const hasDetail = step.runs > 0

  return (
    <li className="relative flex gap-3 pb-4 last:pb-0">
      {!last && <span aria-hidden className="absolute top-7 bottom-0 left-3 w-px bg-border" />}
      <StepIcon status={step.status} />
      <div className="min-w-0 flex-1">
        <button
          type="button"
          className="flex w-full items-start justify-between gap-2 rounded-md text-left outline-none focus-visible:ring-3 focus-visible:ring-ring disabled:cursor-default"
          aria-expanded={hasDetail ? open : undefined}
          aria-controls={hasDetail ? detailId : undefined}
          disabled={!hasDetail}
          onClick={() => setOpen((o) => !o)}
        >
          <span className="min-w-0">
            <span
              className={cn(
                'flex flex-wrap items-center gap-x-2 text-body-sm font-bold',
                step.status === 'pending' ? 'text-muted-foreground' : 'text-ink',
                step.status === 'running' && 'text-primary',
              )}
            >
              {STAGE_LABELS[step.stage]}
              {step.runs > 1 && (
                <span className="rounded-full bg-accent-soft px-2 py-0.5 text-body-xs whitespace-nowrap text-info">
                  round {step.runs}
                </span>
              )}
              <span className="sr-only">, {step.status}</span>
            </span>
            {step.status === 'failed' && step.error ? (
              <span className="block text-body-xs break-words text-error">{step.error}</span>
            ) : (
              step.summary && (
                <span className="block truncate text-body-xs text-muted-foreground">{step.summary}</span>
              )
            )}
          </span>
          <span className="flex shrink-0 items-center gap-1 text-body-xs text-muted-foreground">
            {duration !== null && formatDuration(duration)}
            {hasDetail && (
              <ChevronDown aria-hidden className={cn('size-4 transition-transform', open && 'rotate-180')} />
            )}
          </span>
        </button>
        {hasDetail && open && (
          <dl
            id={detailId}
            className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-body-xs text-muted-foreground"
          >
            <dt>Stage</dt>
            <dd className="font-mono">{step.stage}</dd>
            {step.startedAt && (
              <>
                <dt>Started</dt>
                <dd>{formatDateTime(step.startedAt)}</dd>
              </>
            )}
            {step.runs > 1 && (
              <>
                <dt>Runs</dt>
                <dd>{step.runs} (the latest run is shown)</dd>
              </>
            )}
            {step.summary && (
              <>
                <dt>Summary</dt>
                <dd className="break-words">{step.summary}</dd>
              </>
            )}
          </dl>
        )}
      </div>
    </li>
  )
}

function StepIcon({ status }: { status: StepStatus }) {
  const base = 'relative z-10 flex size-6 shrink-0 items-center justify-center rounded-full'
  switch (status) {
    case 'pending':
      return (
        <span className={cn(base, 'bg-card text-subtle')}>
          <Circle aria-hidden className="size-4" />
        </span>
      )
    case 'running':
      return (
        <span className={cn(base, 'bg-primary text-primary-foreground')}>
          <Loader2 aria-hidden className="size-4 animate-spin" />
        </span>
      )
    case 'finished':
      return (
        <span className={cn(base, 'bg-success-soft text-success')}>
          <Check aria-hidden className="size-4" />
        </span>
      )
    case 'failed':
      return (
        <span className={cn(base, 'bg-error-soft text-error')}>
          <X aria-hidden className="size-4" />
        </span>
      )
  }
}
