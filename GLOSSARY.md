# KOL Matcher

KOL Matcher gives a brand a ranked, explained shortlist of Thai TikTok accounts that fit it.

## Language

### People

**Marketer**:
The person who uses the app to choose KOLs for a brand. Knows terms such as follower and engagement, but not how the system works.
_Avoid_: User, operator

**KOL**:
A Thai TikTok account that the app considers or recommends for a brand. The Thai interface also writes this as "KOL".
_Avoid_: Creator, influencer, ครีเอเตอร์, อินฟลูเอนเซอร์

**Seller account** (บัญชีผู้ขาย):
A TikTok account whose main purpose is to sell or promote one business's goods or services, whether that business is its own or another's. This covers dealers, sales staff, shops, and any business's official account. It is never a KOL, but someone who makes content people follow for its own sake and also earns from selling remains a KOL.
_Avoid_: Shop account, ร้านค้า, ตัวแทน

### Finding KOLs

**Search** (การค้นหา):
One request from a Marketer, from the brand's links through to its recommended KOLs.
_Avoid_: Run, job, pipeline

**TikTok query** (คำค้นหา):
One search phrase sent to TikTok while looking for KOLs. A Search uses many of them.
_Avoid_: Search

**KOL tier** (ขนาด KOL):
The follower-size group of KOLs a brand should work with: Nano, Micro, Mid, or Macro.
_Avoid_: Brand tier

**Brand research** (ค้นข้อมูลแบรนด์):
Web searches the LLM runs to learn what the brand's own pages leave out. Its notes feed the brand analysis.
_Avoid_: Search, web search (for the step as a whole)

### Judging a KOL

**Match score** (ความเหมาะสมรวม):
How well a KOL suits the brand overall, shown out of 100. It is a ranking score, not a chance of campaign success.
_Avoid_: Final, percent match

**Brand fit** (ตรงกับแบรนด์):
How closely a KOL's content matches the brand.
_Avoid_: F

**Engagement** (การมีส่วนร่วม):
How actively viewers respond to a KOL's videos, compared with KOLs of similar size.
_Avoid_: E

**Size fit** (ขนาดผู้ติดตามเหมาะสม):
How close a KOL's follower count is to the brand's KOL tier.
_Avoid_: Tier fit, T

**Audience fit** (กลุ่มผู้ชม):
An estimate of how well a KOL's viewers match the brand's target audience, read from text rather than real viewer data.
_Avoid_: A_eff, demographics

**Style fit** (สไตล์เข้ากับแบรนด์):
How well the way a KOL makes content suits the brand, read from captions and hashtags. Shown to the Marketer only; it is not part of the Match score.
_Avoid_: Brand fit, tone score

**Limited data** (ข้อมูลน้อย):
A mark on a score the system has too little evidence to trust fully.
_Avoid_: Confidence, conf

**Warning label** (ป้ายเตือน):
A short note about something a Marketer should check before choosing a KOL, such as a recent sponsored post for a competitor.
_Avoid_: Flag

**Sample data** (ข้อมูลตัวอย่าง):
Made-up data shown in place of real brand or TikTok data, which a Marketer must not use for real decisions.
_Avoid_: Mock, demo data
