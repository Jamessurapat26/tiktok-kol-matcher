export type FieldErrors = Partial<Record<'website_url' | 'facebook_url' | 'top_k', string>>

function httpUrl(value: string): URL | null {
  try {
    const url = new URL(value.trim())
    return url.protocol === 'http:' || url.protocol === 'https:' ? url : null
  } catch {
    return null
  }
}

/** Client-side checks that mirror the API's JobCreate validators. */
export function validateBrandInput(input: { website: string; facebook: string; topK: string }): FieldErrors {
  const errors: FieldErrors = {}
  const website = httpUrl(input.website)
  if (!website) errors.website_url = 'Enter a full URL, e.g. https://example-brand.co.th'
  else if (website.hostname === 'localhost') errors.website_url = 'Use a public website address'

  const facebook = httpUrl(input.facebook)
  const host = facebook?.hostname ?? ''
  if (!facebook) errors.facebook_url = 'Enter a full URL, e.g. https://www.facebook.com/example-brand'
  else if (!(host === 'facebook.com' || host === 'fb.com' || host.endsWith('.facebook.com'))) {
    errors.facebook_url = 'Use a facebook.com or fb.com page'
  }

  const topK = Number(input.topK)
  if (!Number.isInteger(topK) || topK < 1 || topK > 200) errors.top_k = 'Enter a whole number from 1 to 200'
  return errors
}
