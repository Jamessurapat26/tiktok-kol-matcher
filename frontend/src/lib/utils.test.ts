import { cn } from './utils'

describe('cn', () => {
  it('keeps a text colour next to a custom text size', () => {
    expect(cn('text-primary-foreground', 'text-body-md')).toBe('text-primary-foreground text-body-md')
  })

  it('lets a custom text size replace a Tailwind one', () => {
    expect(cn('text-sm', 'text-heading-sm')).toBe('text-heading-sm')
  })
})
