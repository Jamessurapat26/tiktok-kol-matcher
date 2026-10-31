import { History } from 'lucide-react'
import { useHealth } from '@/api/queries'
import type { Health } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

/** Warns whenever results are not all real data. Fully live needs no badge. */
function sampleDataLabel(health: Health): string | null {
  if (health.mode === 'demo') return 'ข้อมูลตัวอย่าง'
  return Object.values(health.defaults).every(Boolean) ? null : 'บางส่วนเป็นข้อมูลตัวอย่าง'
}

export function Header({ onOpenHistory }: { onOpenHistory: () => void }) {
  const health = useHealth()
  const sampleData = health.data ? sampleDataLabel(health.data) : null

  return (
    <header className="flex items-center justify-between gap-4 py-6">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-heading-md text-ink">KOL Matcher</h1>
        {sampleData && <Badge variant="warning">{sampleData}</Badge>}
        {health.isError && <Badge variant="error">เชื่อมต่อระบบไม่ได้</Badge>}
      </div>
      <Button variant="outline" size="lg" onClick={onOpenHistory}>
        <History aria-hidden />
        ประวัติการค้นหา
      </Button>
    </header>
  )
}
