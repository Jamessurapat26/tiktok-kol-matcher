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
        <h2 className="text-heading-md text-ink">Find Thai TikTok creators for a brand</h2>
        <p className="max-w-prose text-body-md text-ink">
          Enter the brand&apos;s website and Facebook page, and the pipeline returns a ranked shortlist of
          creators.
        </p>
        <Button variant="outline" size="lg" onClick={onTryDemo}>
          Try demo
        </Button>
      </CardContent>
    </Card>
  )
}
