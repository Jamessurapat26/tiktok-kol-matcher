import type { ReactNode } from 'react'
import type { BrandProfile } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { formatCompact, formatPercent } from '@/lib/format'

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return url
  }
}

function audienceText({ target_audience: audience }: BrandProfile): string {
  const gender = audience.gender === 'any' ? 'any gender' : audience.gender
  return `${audience.age_range}, ${gender}`
}

function Keywords({ words }: { words: string[] }) {
  if (words.length === 0) return <span className="text-muted-foreground">—</span>
  return (
    <ul className="flex flex-wrap gap-2">
      {words.map((keyword) => (
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
  )
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
            <Item label="Tier" wide>
              <span>
                <strong className="font-bold text-ink">{profile.tier_label}</strong>
                {' · '}
                {formatCompact(profile.target_followers)} target followers
                {' · '}
                {formatPercent(profile.tier_confidence)} confidence
              </span>
              <span className="mt-1 block text-muted-foreground">{profile.tier_reason}</span>
            </Item>
            <Item label="Target audience">
              {audienceText(profile)}
              {profile.target_audience.interests.length > 0 && (
                <span className="block text-muted-foreground">
                  {profile.target_audience.interests.join(', ')}
                </span>
              )}
            </Item>
            <Item label="Tone">{profile.tone}</Item>
            <Item label="Key products" wide>
              <Keywords words={profile.key_products} />
            </Item>
            <Item label="Topic keywords (TH)" wide>
              <Keywords words={profile.topic_keywords_th} />
            </Item>
            <Item label="Topic keywords (EN)" wide>
              <Keywords words={profile.topic_keywords_en} />
            </Item>
            <Item label="Competitors" wide>
              <Keywords words={profile.competitors} />
            </Item>
            {profile.retailers && profile.retailers.length > 0 && (
              <Item label="Retailers" wide>
                <Keywords words={profile.retailers} />
              </Item>
            )}
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
