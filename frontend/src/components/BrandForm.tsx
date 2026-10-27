import { Loader2 } from 'lucide-react'
import { cloneElement, type ComponentProps, type FormEvent, type ReactElement, useId, useState } from 'react'
import { ApiError } from '@/api/client'
import { useCreateJob, useHealth } from '@/api/queries'
import type { LiveFlag, LiveFlags } from '@/api/types'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { type FieldErrors, validateBrandInput } from '@/lib/validation'

export interface BrandDraft {
  website: string
  facebook: string
  topK: string
}

const FLAG_LABELS: Record<LiveFlag, string> = {
  live_web: 'Website reader',
  live_fb: 'Facebook page',
  live_kols: 'TikTok search',
  live_llm: 'LLM',
  live_embed: 'Embeddings',
}

interface Props {
  draft: BrandDraft
  onDraftChange: (draft: BrandDraft) => void
  onCreated: (jobId: string) => void
}

export function BrandForm({ draft, onDraftChange, onCreated }: Props) {
  const health = useHealth()
  const createJob = useCreateJob()
  // Validate live once the user has tried to submit, so errors clear as they are fixed.
  const [submitted, setSubmitted] = useState(false)
  const clientErrors = submitted ? validateBrandInput(draft) : {}
  // In live mode a flag defaults on exactly when it can run, so defaults double as availability.
  const available = health.data?.mode === 'live' ? health.data.defaults : null
  const [flags, setFlags] = useState<Partial<LiveFlags>>({})

  const apiFields: FieldErrors = createJob.error instanceof ApiError ? createJob.error.fields : {}
  const errors: FieldErrors = { ...apiFields, ...clientErrors }
  // A 422 on a field this form shows goes inline; anything else (400, 429, network) goes above the button.
  const inline = Boolean(apiFields.website_url || apiFields.facebook_url || apiFields.top_k)
  const formError = createJob.error && !inline ? createJob.error.message : null

  const set = (field: keyof BrandDraft, value: string) => {
    onDraftChange({ ...draft, [field]: value })
    if (createJob.error) createJob.reset()
  }

  const submit = (event: FormEvent) => {
    event.preventDefault()
    setSubmitted(true)
    if (Object.keys(validateBrandInput(draft)).length) return
    createJob.mutate(
      {
        website_url: draft.website.trim(),
        facebook_url: draft.facebook.trim(),
        top_k: Number(draft.topK),
        ...(available ? flags : {}),
      },
      { onSuccess: ({ job_id }) => onCreated(job_id) },
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Brand</CardTitle>
        <CardDescription>Where the brand lives online.</CardDescription>
      </CardHeader>
      <CardContent>
        <form className="flex flex-col gap-5" onSubmit={submit} noValidate>
          <Field id="website_url" label="Website URL" error={errors.website_url}>
            <Input
              type="url"
              inputMode="url"
              placeholder="https://example-brand.co.th"
              value={draft.website}
              onChange={(e) => set('website', e.target.value)}
            />
          </Field>
          <Field id="facebook_url" label="Facebook URL" error={errors.facebook_url}>
            <Input
              type="url"
              inputMode="url"
              placeholder="https://www.facebook.com/example-brand"
              value={draft.facebook}
              onChange={(e) => set('facebook', e.target.value)}
            />
          </Field>
          <Field id="top_k" label="Top K" error={errors.top_k}>
            <Input
              type="number"
              min={1}
              max={200}
              className="w-28"
              value={draft.topK}
              onChange={(e) => set('topK', e.target.value)}
            />
          </Field>

          {available && (
            <fieldset className="flex flex-col gap-3">
              <legend className="mb-2 text-body-sm font-bold text-ink">Run live</legend>
              {(Object.keys(FLAG_LABELS) as LiveFlag[]).map((flag) => (
                <FlagToggle
                  key={flag}
                  flag={flag}
                  disabled={!available[flag]}
                  checked={flags[flag] ?? available[flag]}
                  onChange={(on) => setFlags((f) => ({ ...f, [flag]: on }))}
                />
              ))}
            </fieldset>
          )}

          {formError && (
            <p role="alert" className="rounded-lg bg-error-soft px-3 py-2 text-body-sm text-error">
              {formError}
            </p>
          )}

          <Button
            type="submit"
            className="h-11.5 text-body-md font-bold shadow-cta"
            disabled={createJob.isPending}
          >
            {createJob.isPending && <Loader2 className="animate-spin" aria-hidden />}
            Find KOLs
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}

function Field({
  id,
  label,
  error,
  children,
}: {
  id: string
  label: string
  error?: string
  children: ReactElement<ComponentProps<'input'>>
}) {
  const errorId = `${id}-error`
  const input = cloneElement(children, {
    id,
    'aria-invalid': !!error,
    'aria-describedby': error ? errorId : undefined,
  })
  return (
    <div className="flex flex-col gap-2">
      <Label htmlFor={id} className="text-body-sm font-bold text-ink">
        {label}
      </Label>
      {input}
      {error && (
        <p id={errorId} role="alert" className="text-body-xs text-error">
          {error}
        </p>
      )}
    </div>
  )
}

function FlagToggle({
  flag,
  checked,
  disabled,
  onChange,
}: {
  flag: LiveFlag
  checked: boolean
  disabled: boolean
  onChange: (on: boolean) => void
}) {
  const id = useId()
  return (
    <div className="flex items-center justify-between gap-3">
      <Label htmlFor={id} className="text-body-sm text-ink">
        {FLAG_LABELS[flag]}
        {disabled && <span className="text-muted-foreground"> (unavailable)</span>}
      </Label>
      <Switch id={id} checked={checked && !disabled} disabled={disabled} onCheckedChange={onChange} />
    </div>
  )
}
