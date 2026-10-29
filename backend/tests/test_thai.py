from app.models import KolProfile
from app.thai import has_thai_script, is_lao_creator, is_thai_creator


def _kol(handle: str = "creator", **overrides) -> KolProfile:
    return KolProfile(
        handle=handle,
        url=f"https://www.tiktok.com/@{handle}",
        source="apify",
        **overrides,
    )


def test_thai_language_is_kept():
    kol = _kol(language="TH", region=None, bio="English bio.")

    assert is_thai_creator(kol)


def test_thai_region_is_kept():
    kol = _kol(language="en", region="th", bio="English bio.")

    assert is_thai_creator(kol)


def test_thai_script_bio_is_kept_without_language_or_region():
    kol = _kol(language="en", region=None, bio="รีวิวแกดเจ็ต")

    assert is_thai_creator(kol)


def test_thai_script_nickname_is_kept():
    kol = _kol(nickname="พลอยรีวิว", language="en", region=None)

    assert is_thai_creator(kol)


def test_thai_script_search_caption_is_kept():
    kol = _kol(
        language="en",
        region=None,
        bio="English bio.",
        content_summary="แว่นกันแสงสำหรับเกมเมอร์",
    )

    assert is_thai_creator(kol)


def test_english_only_us_creator_is_dropped():
    kol = _kol(
        language="en",
        region="US",
        nickname="Gadget Gary",
        bio="Hands-on tech reviews.",
        content_summary="Desk setup tour",
    )

    assert not is_thai_creator(kol)


def test_unknown_language_and_region_with_english_bio_is_dropped():
    kol = _kol(language=None, region=None, bio="English bio only.")

    assert not is_thai_creator(kol)


def test_a_single_stray_thai_character_is_not_enough():
    kol = _kol(language=None, region=None, bio="hello ก")

    assert not has_thai_script(kol.bio)
    assert not is_thai_creator(kol)


def test_lao_script_creator_tagged_as_thai_is_not_thai():
    # TikTok often tags Lao text as Thai; the script itself decides.
    kol = _kol(
        nickname="ນ້ອງແກ້ວ",
        bio="ຣີວິວເຄື່ອງສຳອາງ ສົ່ງທົ່ວລາວ",
        language="th",
        region=None,
    )

    assert is_lao_creator(kol)
    assert not is_thai_creator(kol)


def test_lao_region_without_thai_text_is_not_thai():
    kol = _kol(language="en", region="LA", bio="Beauty reviews")

    assert not is_thai_creator(kol)


def test_mostly_thai_text_outweighs_a_lao_region():
    kol = _kol(language="th", region="LA", bio="รีวิวเครื่องสำอาง ส่งทั่วไทย")

    assert not is_lao_creator(kol)
    assert is_thai_creator(kol)


def test_a_few_lao_characters_do_not_reject_a_thai_creator():
    kol = _kol(language="th", region="TH", bio="รีวิวเครื่องสำอางทุกวัน ສະບາຍດີ")

    assert is_thai_creator(kol)
