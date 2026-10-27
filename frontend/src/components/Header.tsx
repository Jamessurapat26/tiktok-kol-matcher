import { History } from 'lucide-react'
import { useHealth } from '@/api/queries'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

export function Header({ onOpenHistory }: { onOpenHistory: () => void }) {
  const health = useHealth()
  const mode = health.data?.mode

  return (
    <header className="flex items-center justify-between gap-4 py-6">
      <div className="flex items-center gap-3">
        <h1 className="text-heading-md text-ink">KOL Matcher</h1>
        {mode && (
          <Badge variant={mode === 'live' ? 'success' : 'info'} aria-label={`Mode: ${mode}`}>
            {mode === 'live' ? 'Live' : 'Demo'}
          </Badge>
        )}
        {health.isError && <Badge variant="error">API offline</Badge>}
      </div>
      <Button variant="outline" size="lg" onClick={onOpenHistory}>
        <History aria-hidden />
        History
      </Button>
    </header>
  )
}
