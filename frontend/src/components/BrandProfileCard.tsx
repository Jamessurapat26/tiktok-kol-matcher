import type { ReactNode } from 'react'
import type { BrandProfile } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return url
  }
}

function audienceText({ target_audience: a }: BrandProfile): string {
  const ages =
    a.age_min !== null || a.age_max !== null ? `${a.age_min ?? '?'}–${a.age_max ?? '?'}` : 'any age'
  const gender = a.gender === 'any' ? 'any gender' : a.gender
  return `${ages}, ${gender}, ${a.location}`
}

interface Props {
  websiteUrl: string | null
  profile: BrandProfile | null
  /** The pipeline stopped before a profile was made. */
  unavailable: boolean
}

export function BrandProfileCard({ websiteUrl, profile, unavailable }: Props) {
  // The API has no brand name field; the website host names the brand.
  const name = websiteUrl ? hostOf(websiteUrl) : 'Brand'
  return (
    <Card aria-busy={!profile && !unavailable}>
      <CardHeader>
        <CardTitle className="text-heading-md">{name}</CardTitle>
        <CardDescription>{profile ? profile.category : 'Brand profile'}</CardDescription>
      </CardHeader>
      <CardContent>
        {profile ? (
          <dl className="grid gap-5 sm:grid-cols-2">
            <Item label="Target audience">
              {audienceText(profile)}
              {profile.target_audience.interests.length > 0 && (
                <span className="block text-muted-foreground">
                  {profile.target_audience.interests.join(', ')}
                </span>
              )}
            </Item>
            <Item label="Tone">{profile.tone}</Item>
            <Item label="Keywords" wide>
              <ul className="flex flex-wrap gap-2">
                {[...new Set([...profile.keywords_th, ...profile.keywords_en])].map((keyword) => (
                  <li key={keyword} className="min-w-0">
                    <Badge
                      variant="secondary"
                      className="h-auto max-w-full px-3 py-1 text-left text-body-xs whitespace-normal wrap-anywhere"
                    >
                      {keyword}
                    </Badge>
                  </li>
                ))}
              </ul>
            </Item>
          </dl>
        ) : unavailable ? (
          <p className="text-body-sm text-muted-foreground">
            No brand profile: the run stopped before brand analysis finished.
          </p>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2">
            <span className="sr-only">Loading brand profile</span>
            <Skeleton className="h-12" />
            <Skeleton className="h-12" />
            <Skeleton className="h-8 sm:col-span-2" />
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function Item({ label, wide, children }: { label: string; wide?: boolean; children: ReactNode }) {
  return (
    <div className={wide ? 'sm:col-span-2' : undefined}>
      <dt className="mb-1 text-body-xs font-bold tracking-wide text-muted-foreground uppercase">{label}</dt>
      <dd className="text-body-sm text-ink">{children}</dd>
    </div>
  )
}
