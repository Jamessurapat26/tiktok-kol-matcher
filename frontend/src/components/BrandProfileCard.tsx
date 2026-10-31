import type { ReactNode } from 'react'
import type { BrandProfile, TargetAudience, Tier } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { formatCompact, hostOf } from '@/lib/format'

const GENDER_LABELS: Record<TargetAudience['gender'], string> = {
  any: 'ทุกเพศ',
  male: 'ผู้ชาย',
  female: 'ผู้หญิง',
}

const TIER_LABELS: Record<Tier, string> = { nano: 'Nano', micro: 'Micro', mid: 'Mid', macro: 'Macro' }

function audienceText({ target_audience: audience }: BrandProfile): string {
  const age = audience.age_range === 'any' ? 'ทุกวัย' : `อายุ ${audience.age_range}`
  return `${age}, ${GENDER_LABELS[audience.gender]}`
}

/** The Thai display text, or the original on briefs stored before it existed. */
function thai(text: string | undefined, original: string): string
function thai(text: string[] | undefined, original: string[]): string[]
function thai(text: string | string[] | undefined, original: string | string[]) {
  return text?.length ? text : original
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
  /** The search stopped before a profile was made. */
  unavailable: boolean
}

export function BrandProfileCard({ websiteUrl, profile, unavailable }: Props) {
  // The API has no brand name field; the website host names the brand.
  const name = websiteUrl ? hostOf(websiteUrl) : 'แบรนด์'
  const interests = profile ? thai(profile.interests_th, profile.target_audience.interests) : []
  const keywords = profile ? [...new Set([...profile.topic_keywords_th, ...profile.topic_keywords_en])] : []
  return (
    <Card aria-busy={!profile && !unavailable}>
      <CardHeader>
        <CardTitle className="text-heading-md">{name}</CardTitle>
        <CardDescription>
          {profile ? thai(profile.category_th, profile.category) : 'ข้อมูลแบรนด์'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {profile ? (
          <dl className="grid gap-5 sm:grid-cols-2">
            <Item label="ขนาด KOL ที่แนะนำ" wide>
              <span>
                <strong className="font-bold text-ink">{TIER_LABELS[profile.tier_label]}</strong>
                {' · '}
                ผู้ติดตามราว {formatCompact(profile.target_followers)}
              </span>
              <span className="mt-1 block text-muted-foreground">
                {thai(profile.tier_reason_th, profile.tier_reason)}
              </span>
            </Item>
            <Item label="กลุ่มเป้าหมาย">
              {audienceText(profile)}
              {interests.length > 0 && (
                <span className="block text-muted-foreground">{interests.join(', ')}</span>
              )}
            </Item>
            <Item label="โทนการสื่อสาร">{thai(profile.tone_th, profile.tone)}</Item>
            <Item label="สินค้าหลัก" wide>
              <Keywords words={thai(profile.key_products_th, profile.key_products)} />
            </Item>
            <Item label="คำค้นหาที่เกี่ยวข้อง" wide>
              <Keywords words={keywords} />
            </Item>
            <Item label="คู่แข่ง" wide>
              <Keywords words={profile.competitors} />
            </Item>
            {profile.retailers && profile.retailers.length > 0 && (
              <Item label="ร้านค้าที่จำหน่าย" wide>
                <Keywords words={profile.retailers} />
              </Item>
            )}
          </dl>
        ) : unavailable ? (
          <p className="text-body-sm text-muted-foreground">
            ไม่มีข้อมูลแบรนด์ เพราะการค้นหาหยุดก่อนวิเคราะห์แบรนด์เสร็จ
          </p>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2">
            <span className="sr-only">กำลังโหลดข้อมูลแบรนด์</span>
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
      <dt className="mb-1 text-body-xs font-bold text-muted-foreground">{label}</dt>
      <dd className="text-body-sm text-ink">{children}</dd>
    </div>
  )
}
