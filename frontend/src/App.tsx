import { useState } from 'react'
import { useJob } from '@/api/queries'
import type { MatchRequest } from '@/api/types'
import { type BrandDraft, BrandForm } from '@/components/BrandForm'
import { BrandProfileCard } from '@/components/BrandProfileCard'
import { EmptyState } from '@/components/EmptyState'
import { Header } from '@/components/Header'
import { HistorySheet } from '@/components/HistorySheet'
import { ProgressCard } from '@/components/ProgressCard'
import { ResultsCard } from '@/components/ResultsCard'
import { useJobParam } from '@/hooks/useJobParam'
import { useJobProgress } from '@/hooks/useJobProgress'

const EMPTY_DRAFT: BrandDraft = { website: '', facebook: '', topK: '10' }
const DEMO_DRAFT: BrandDraft = {
  website: 'https://example-brand.co.th',
  facebook: 'https://www.facebook.com/example-brand',
  topK: '10',
}

function toDraft(request: MatchRequest): BrandDraft {
  return { website: request.website_url, facebook: request.facebook_url, topK: String(request.top_k) }
}

export default function App() {
  const [jobId, setJobId] = useJobParam()
  const [historyOpen, setHistoryOpen] = useState(false)
  const { progress, connected } = useJobProgress(jobId)
  const job = useJob(jobId)

  // Edits belong to the job they were typed under. A job loaded from ?job= or
  // History shows its own request until the user types again.
  const [edits, setEdits] = useState<{ jobId: string | null; draft: BrandDraft }>({
    jobId: null,
    draft: EMPTY_DRAFT,
  })
  const request = job.data?.request
  const draft = edits.jobId === jobId || !request ? edits.draft : toDraft(request)
  const setDraft = (next: BrandDraft) => setEdits({ jobId, draft: next })

  // The stream refetches the job on its terminal event, so the job's own status is the source of truth.
  const status = job.data?.status
  const terminal = status === 'succeeded' || status === 'failed'

  return (
    <div className="mx-auto max-w-7xl px-4 pb-section sm:px-6 lg:px-8">
      <Header onOpenHistory={() => setHistoryOpen(true)} />
      <main className="grid items-start gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
        <div className="flex flex-col gap-6">
          <BrandForm draft={draft} onDraftChange={setDraft} onCreated={setJobId} />
          {jobId && !job.isError && <ProgressCard progress={progress} connected={connected} />}
        </div>
        <div className="flex flex-col gap-6">
          {!jobId ? (
            <EmptyState onTryDemo={() => setDraft(DEMO_DRAFT)} />
          ) : job.isError ? (
            <p role="alert" className="rounded-xl bg-error-soft p-4 text-body-sm text-error">
              Could not load job {jobId}: {job.error.message}
            </p>
          ) : (
            <>
              <BrandProfileCard
                websiteUrl={job.data?.request.website_url ?? null}
                profile={job.data?.brand_profile ?? null}
                unavailable={terminal && !job.data?.brand_profile}
              />
              <ResultsCard
                jobId={jobId}
                results={job.data?.results ?? []}
                pending={!terminal}
                failed={status === 'failed'}
              />
            </>
          )}
        </div>
      </main>
      <HistorySheet open={historyOpen} onOpenChange={setHistoryOpen} selectedId={jobId} onSelect={setJobId} />
    </div>
  )
}
