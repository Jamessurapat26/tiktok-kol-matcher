import { fireEvent, render, screen } from '@testing-library/react'
import { BrandProfileCard } from './BrandProfileCard'

const WEBSITE = 'https://www.example-brand.co.th/'
const LOGO_NAME = 'โลโก้ example-brand.co.th'

function card(logoUrl: string | null) {
  return <BrandProfileCard websiteUrl={WEBSITE} logoUrl={logoUrl} profile={null} unavailable={false} />
}

describe('BrandProfileCard logo', () => {
  it('renders the logo when logoUrl is set', () => {
    render(card('https://example.com/logo.jpg'))

    const img = screen.getByRole('img', { name: LOGO_NAME })
    expect(img).toHaveAttribute('src', 'https://example.com/logo.jpg')
    expect(img).toHaveAttribute('referrerpolicy', 'no-referrer')
    expect(screen.queryByTestId('brand-initials')).not.toBeInTheDocument()
  })

  it('shows initials when there is no logo', () => {
    render(card(null))

    expect(screen.queryByRole('img', { name: LOGO_NAME })).not.toBeInTheDocument()
    expect(screen.getByTestId('brand-initials')).toHaveTextContent('EX')
  })

  it('falls back to initials when the image fails, and retries for a new logo', () => {
    const { rerender } = render(card('https://example.com/expired.jpg'))

    fireEvent.error(screen.getByRole('img', { name: LOGO_NAME }))
    expect(screen.getByTestId('brand-initials')).toBeInTheDocument()

    rerender(card('https://example.com/other.jpg'))
    expect(screen.getByRole('img', { name: LOGO_NAME })).toHaveAttribute(
      'src',
      'https://example.com/other.jpg',
    )
  })
})
