import { createCn } from 'cn/config'
import tokens from '@/styles/tokens.css?raw'

// The merge tables know Tailwind's own text sizes only, so a `--text-<token>`
// size such as text-body-md would otherwise count as a text colour and knock
// out whichever colour sits next to it. The list comes from the token file.
const textSizes = [...tokens.matchAll(/--text-([a-z0-9]+(?:-[a-z0-9]+)*):/g)].map((m) => m[1])

export const cn = createCn({ extend: { classGroups: { 'font-size': [{ text: textSizes }] } } })
