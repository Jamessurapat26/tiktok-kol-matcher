import { Check, Circle, Loader2, X } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { failureMessage, type Progress, type StepStatus, type VisibleStep, visibleSteps } from '@/lib/steps'
import { cn } from '@/lib/utils'

const STATUS_LABELS: Record<StepStatus, string> = {
  pending: 'รอดำเนินการ',
  running: 'กำลังทำงาน',
  finished: 'เสร็จแล้ว',
  failed: 'ไม่สำเร็จ',
}

interface Props {
  progress: Progress
  connected: boolean
  /** How many KOLs the finished Search ranked, once the job has been refetched. */
  resultCount?: number
}

export function ProgressCard({ progress, connected, resultCount }: Props) {
  const { outcome } = progress
  const steps = visibleSteps(progress)
  const running = steps.find((step) => step.status === 'running')
  const status = outcome
    ? outcome.type === 'done'
      ? resultCount === undefined
        ? 'จัดอันดับ KOL เสร็จแล้ว'
        : `จัดอันดับ KOL แล้ว ${resultCount} คน`
      : failureMessage(progress)
    : running
      ? `กำลัง${running.label}`
      : connected
        ? 'กำลังเริ่มการค้นหา'
        : 'กำลังเชื่อมต่อ…'

  return (
    <Card>
      <CardHeader>
        <CardTitle>ความคืบหน้า</CardTitle>
        <CardDescription aria-live="polite" className={cn(outcome?.type === 'failed' && 'text-error')}>
          {status}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ol className="flex flex-col">
          {steps.map((step, i) => (
            <StepRow key={step.label} step={step} last={i === steps.length - 1} />
          ))}
        </ol>
      </CardContent>
    </Card>
  )
}

function StepRow({ step, last }: { step: VisibleStep; last: boolean }) {
  return (
    <li className="relative flex gap-3 pb-4 last:pb-0">
      {!last && <span aria-hidden className="absolute top-7 bottom-0 left-3 w-px bg-border" />}
      <StepIcon status={step.status} />
      <div className="min-w-0 flex-1">
        <span
          className={cn(
            'block text-body-sm font-bold',
            step.status === 'pending' ? 'text-muted-foreground' : 'text-ink',
            step.status === 'running' && 'text-primary',
          )}
        >
          {step.label}
          <span className="sr-only">, {STATUS_LABELS[step.status]}</span>
        </span>
        {step.partial && (
          <span className="block text-body-xs text-warning">ทำได้บางส่วน ระบบทำงานต่อแล้ว</span>
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
