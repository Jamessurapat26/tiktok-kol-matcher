from app.models import MAX_POST_CHARS, MAX_RECENT_POSTS, FacebookPage
from app.prompts import brand_messages


def _page(**overrides) -> FacebookPage:
    data = {
        "page_name": "ExampleBrand",
        "title": "ExampleBrand",
        "url": "https://www.facebook.com/examplebrand",
    }
    data.update(overrides)
    return FacebookPage.model_validate(data)


def test_recent_posts_are_trimmed_clipped_and_capped():
    posts = ["  first\n\npost  ", "   ", "x" * (MAX_POST_CHARS + 50)]
    posts += [f"post {n}" for n in range(30)]

    page = _page(recent_posts=posts)

    assert len(page.recent_posts) == MAX_RECENT_POSTS
    assert page.recent_posts[0] == "first post"
    assert len(page.recent_posts[1]) == MAX_POST_CHARS
    assert _page().recent_posts == []


def test_brand_prompt_includes_recent_posts_and_website():
    page = _page(
        website="https://example-brand.co.th",
        recent_posts=["New blue-light frames are in stock", "y" * 2_000],
    )

    user = brand_messages("site text", page)[1]["content"]

    assert "website: https://example-brand.co.th" in user
    assert "recent_posts:\n- New blue-light frames are in stock" in user
    assert "y" * 400 not in user
