import { fireEvent, render, screen } from '@testing-library/react'
import type { RankedKol } from '@/api/types'
import { initials } from '@/lib/format'
import { ResultsCard } from './ResultsCard'

function ranked(handle: string, avatarUrl?: string | null): RankedKol {
  return {
    kol: {
      handle,
      url: `https://www.tiktok.com/@${handle}`,
      nickname: null,
      ...(avatarUrl === undefined ? {} : { avatar_url: avatarUrl }),
      followers: 1000,
      bio: null,
      language: 'th',
      region: 'TH',
      posts_per_week: 2,
      content_summary: null,
      videos: [],
      stats_source: 'search',
      source: 'seed',
    },
    scores: {
      F: 0.5,
      E: 0.25,
      T: 1,
      A_eff: 0.75,
      F_conf: 0.9,
      E_conf: 0.8,
      T_conf: 0.7,
      A_conf: 0.6,
      sim: 0.4,
      on_topic: 0.3,
      judge: 0.5,
      er_star: 0.1,
      reach: 0.2,
      judged: true,
    },
    quality: 0.5,
    final: 0.5,
    flags: [],
    reason: 'Good fit.',
    evidence_urls: [],
    profile_url: `https://www.tiktok.com/@${handle}`,
  }
}

function renderCard(row: RankedKol) {
  render(<ResultsCard websiteUrl={null} results={[row]} pending={false} failed={false} />)
}

describe('ResultsCard avatar', () => {
  it('renders the profile picture when avatar_url is set', () => {
    renderCard(ranked('avatarkol', 'https://example.com/a.jpg'))

    const img = screen.getByRole('img', { name: 'avatarkol' })
    expect(img).toHaveAttribute('src', 'https://example.com/a.jpg')
    expect(img).toHaveAttribute('loading', 'lazy')
    expect(img).toHaveAttribute('referrerpolicy', 'no-referrer')
    expect(screen.queryByTestId('kol-initials')).not.toBeInTheDocument()
  })

  it('falls back to initials when the picture fails to load', () => {
    renderCard(ranked('brokenkol', 'https://example.com/a.jpg'))

    fireEvent.error(screen.getByRole('img', { name: 'brokenkol' }))

    expect(screen.queryByRole('img', { name: 'brokenkol' })).not.toBeInTheDocument()
    expect(screen.getByTestId('kol-initials')).toHaveTextContent(initials('brokenkol'))
  })

  it.each([undefined, null])('shows initials when avatar_url is %s', (avatarUrl) => {
    renderCard(ranked('plainkol', avatarUrl))

    expect(screen.queryByRole('img', { name: 'plainkol' })).not.toBeInTheDocument()
    expect(screen.getByTestId('kol-initials')).toHaveTextContent(initials('plainkol'))
  })
})

describe('content style', () => {
  function renderRow(row: RankedKol) {
    render(<ResultsCard websiteUrl={null} results={[row]} pending={false} failed={false} />)
  }

  it('shows formats, style fit, and its evidence, and says it is not in the score', () => {
    renderRow({
      ...ranked('styled'),
      content: {
        formats: ['review', 'tutorial'],
        style: 'รีวิวจากการใช้งานจริง',
        style_fit: 0.8,
        style_reason: 'รูปแบบตรงกับแบรนด์',
        evidence_urls: ['https://www.tiktok.com/@styled/video/1'],
      },
    })

    expect(screen.getByText('รีวิว')).toBeTruthy()
    expect(screen.getByText('สอน/แนะนำวิธี')).toBeTruthy()
    expect(screen.getByText('รีวิวจากการใช้งานจริง')).toBeTruthy()
    expect(screen.getByText('สไตล์เข้ากับแบรนด์ 80/100')).toBeTruthy()
    expect(screen.getByText(/ยังไม่นับรวมในความเหมาะสมรวม/)).toBeTruthy()
    expect(screen.getByRole('link', { name: /คลิปอ้างอิงสไตล์ 1/ }).getAttribute('href')).toBe(
      'https://www.tiktok.com/@styled/video/1',
    )
  })

  it('counts TikTok ad labels on the checked videos', () => {
    const row = ranked('ads')
    const video = { url: null, views: 1, likes: 0, comments: 0, shares: 0, posted_at: null, caption: 'x' }
    renderRow({ ...row, kol: { ...row.kol, videos: [{ ...video, is_ad: true }, video] } })

    expect(screen.getByText('ติดป้ายโฆษณา 1 จาก 2 คลิปที่ตรวจ')).toBeTruthy()
  })

  it('hides the section when there is nothing to show, as on older stored rows', () => {
    renderRow(ranked('plain'))

    expect(screen.queryByRole('region', { name: 'สไตล์คอนเทนต์' })).toBeNull()
  })
})

describe('ResultsCard seller accounts', () => {
  const props = { websiteUrl: null, pending: false, failed: false }

  it('says how many seller accounts were removed', () => {
    render(<ResultsCard {...props} results={[ranked('a')]} sellersRemoved={14} requested={1} />)

    expect(screen.getByText('ตัดบัญชีผู้ขายออก 14 บัญชี')).toBeInTheDocument()
  })

  it('explains a list shorter than the Marketer asked for', () => {
    render(<ResultsCard {...props} results={[ranked('a')]} sellersRemoved={14} requested={10} />)

    expect(screen.getByText('ตัดบัญชีผู้ขายออก 14 บัญชี จึงแสดงเท่าที่เหลือ 1 คน')).toBeInTheDocument()
  })

  it('explains an empty list when only sellers were found', () => {
    render(<ResultsCard {...props} results={[]} sellersRemoved={3} requested={10} />)

    expect(screen.getByText('ไม่เหลือ KOL ที่แนะนำหลังตัดบัญชีผู้ขายออก 3 บัญชี')).toBeInTheDocument()
  })

  it('stays quiet when no seller was removed', () => {
    render(<ResultsCard {...props} results={[ranked('a')]} requested={10} />)

    expect(screen.queryByText(/ตัดบัญชีผู้ขาย/)).not.toBeInTheDocument()
  })
})
