import { Loader2 } from 'lucide-react'
import { cloneElement, type ComponentProps, type FormEvent, type ReactElement, useState } from 'react'
import { ApiError, friendlyError } from '@/api/client'
import { useCreateJob } from '@/api/queries'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { type FieldErrors, validateBrandInput } from '@/lib/validation'

export interface BrandDraft {
  website: string
  facebook: string
  topK: string
}

// The API's field messages are English, so a rejected field gets this instead.
const API_FIELD_ERROR = 'กรุณาตรวจสอบข้อมูลในช่องนี้'

interface Props {
  draft: BrandDraft
  onDraftChange: (draft: BrandDraft) => void
  onCreated: (jobId: string) => void
}

export function BrandForm({ draft, onDraftChange, onCreated }: Props) {
  const createJob = useCreateJob()
  // Validate live once the user has tried to submit, so errors clear as they are fixed.
  const [submitted, setSubmitted] = useState(false)
  const clientErrors = submitted ? validateBrandInput(draft) : {}

  const apiFields: FieldErrors = createJob.error instanceof ApiError ? createJob.error.fields : {}
  const errors: FieldErrors = {
    ...Object.fromEntries(Object.keys(apiFields).map((field) => [field, API_FIELD_ERROR])),
    ...clientErrors,
  }
  // A 422 on a field this form shows goes inline; anything else (400, 429, network) goes above the button.
  const inline = Boolean(apiFields.website_url || apiFields.facebook_url || apiFields.top_k)
  const formError = createJob.error && !inline ? friendlyError(createJob.error) : null

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
      },
      { onSuccess: ({ job_id }) => onCreated(job_id) },
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>ข้อมูลแบรนด์</CardTitle>
        <CardDescription>ใส่ลิงก์เว็บไซต์และเพจของแบรนด์</CardDescription>
      </CardHeader>
      <CardContent>
        <form className="flex flex-col gap-5" onSubmit={submit} noValidate>
          <Field id="website_url" label="เว็บไซต์ของแบรนด์" error={errors.website_url}>
            <Input
              type="url"
              inputMode="url"
              placeholder="https://example-brand.co.th"
              value={draft.website}
              onChange={(e) => set('website', e.target.value)}
            />
          </Field>
          <Field id="facebook_url" label="เพจ Facebook ของแบรนด์" error={errors.facebook_url}>
            <Input
              type="url"
              inputMode="url"
              placeholder="https://www.facebook.com/example-brand"
              value={draft.facebook}
              onChange={(e) => set('facebook', e.target.value)}
            />
          </Field>
          <Field id="top_k" label="จำนวน KOL ที่ต้องการ" error={errors.top_k}>
            <Input
              type="number"
              min={1}
              max={200}
              className="w-28"
              value={draft.topK}
              onChange={(e) => set('topK', e.target.value)}
            />
          </Field>

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
            ค้นหา KOL
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
