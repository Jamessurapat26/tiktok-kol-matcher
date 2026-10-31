import { Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'

export function EmptyState({ onTryDemo }: { onTryDemo: () => void }) {
  return (
    <Card>
      <CardContent className="flex flex-col items-start gap-4 py-10">
        <span className="flex size-12 items-center justify-center rounded-full bg-accent-soft text-info">
          <Sparkles aria-hidden />
        </span>
        <h2 className="text-heading-md text-ink">ค้นหา KOL ไทยบน TikTok ที่เหมาะกับแบรนด์</h2>
        <p className="max-w-prose text-body-md text-ink">
          ใส่ลิงก์เว็บไซต์และเพจ Facebook ของแบรนด์ แล้วระบบจะจัดอันดับ KOL ที่เหมาะสมพร้อมเหตุผล
        </p>
        <Button variant="outline" size="lg" onClick={onTryDemo}>
          ลองดูตัวอย่าง
        </Button>
      </CardContent>
    </Card>
  )
}
