import { useJobs } from '@/api/queries'
import type { JobStatus } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { Skeleton } from '@/components/ui/skeleton'
import { formatDateTime } from '@/lib/format'
import { cn } from '@/lib/utils'

const STATUS_LABELS: Record<JobStatus, string> = {
  queued: 'รอคิว',
  running: 'กำลังค้นหา',
  succeeded: 'เสร็จแล้ว',
  failed: 'ไม่สำเร็จ',
}

const STATUS_BADGE: Record<JobStatus, 'success' | 'warning' | 'error' | 'info'> = {
  queued: 'warning',
  running: 'info',
  succeeded: 'success',
  failed: 'error',
}

interface Props {
  open: boolean
  onOpenChange: (open: boolean) => void
  selectedId: string | null
  onSelect: (jobId: string) => void
}

export function HistorySheet({ open, onOpenChange, selectedId, onSelect }: Props) {
  const jobs = useJobs(open)

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full gap-0 sm:max-w-md">
        <SheetHeader className="p-6">
          <SheetTitle className="text-heading-sm text-ink">ประวัติการค้นหา</SheetTitle>
          <SheetDescription>การค้นหาล่าสุด เรียงจากใหม่ไปเก่า</SheetDescription>
        </SheetHeader>
        <div className="flex-1 overflow-y-auto px-6 pb-6">
          {jobs.isPending ? (
            <div className="flex flex-col gap-3">
              {[0, 1, 2].map((i) => (
                <Skeleton key={i} className="h-16" />
              ))}
            </div>
          ) : jobs.isError ? (
            <p role="alert" className="text-body-sm text-error">
              โหลดประวัติการค้นหาไม่สำเร็จ กรุณาลองใหม่อีกครั้ง
            </p>
          ) : jobs.data.length === 0 ? (
            <p className="text-body-sm text-muted-foreground">ยังไม่มีการค้นหา</p>
          ) : (
            <ul className="flex flex-col gap-2">
              {jobs.data.map((job) => (
                <li key={job.id}>
                  <button
                    type="button"
                    aria-current={job.id === selectedId ? 'true' : undefined}
                    onClick={() => {
                      onSelect(job.id)
                      onOpenChange(false)
                    }}
                    className={cn(
                      'flex w-full flex-col gap-1 rounded-xl border p-4 text-left outline-none hover:bg-surface focus-visible:ring-3 focus-visible:ring-ring',
                      job.id === selectedId && 'border-ink',
                    )}
                  >
                    <span className="flex items-center justify-between gap-2">
                      <span className="truncate text-body-sm font-bold text-ink">
                        {job.request.website_url}
                      </span>
                      <Badge variant={STATUS_BADGE[job.status]}>{STATUS_LABELS[job.status]}</Badge>
                    </span>
                    <span className="text-body-xs text-muted-foreground">
                      {formatDateTime(job.created_at)} · {job.request.top_k} KOL
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </SheetContent>
    </Sheet>
  )
}
