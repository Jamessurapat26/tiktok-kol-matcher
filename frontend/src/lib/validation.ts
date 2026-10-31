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
  if (!website) errors.website_url = 'กรุณาใส่ลิงก์ให้ครบ เช่น https://example-brand.co.th'
  else if (website.hostname === 'localhost') errors.website_url = 'กรุณาใช้ลิงก์เว็บไซต์ที่เปิดดูได้ทั่วไป'

  const facebook = httpUrl(input.facebook)
  const host = facebook?.hostname ?? ''
  if (!facebook) errors.facebook_url = 'กรุณาใส่ลิงก์ให้ครบ เช่น https://www.facebook.com/example-brand'
  else if (!(host === 'facebook.com' || host === 'fb.com' || host.endsWith('.facebook.com'))) {
    errors.facebook_url = 'กรุณาใช้ลิงก์เพจจาก facebook.com หรือ fb.com'
  }

  const topK = Number(input.topK)
  if (!Number.isInteger(topK) || topK < 1 || topK > 200) errors.top_k = 'กรุณาใส่จำนวนเต็มตั้งแต่ 1 ถึง 200'
  return errors
}
