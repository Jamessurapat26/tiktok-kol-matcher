import json
from dataclasses import replace
from datetime import UTC, datetime

from app.graph import build_graph, initial_state
from app.integrations import mock_integrations
from app.integrations.kol_enricher_live import enrich_from_items
from app.integrations.kol_search_live import map_search_item
from app.integrations.llm import MockBrandAnalyzer
from app.integrations.llm_live import LiveBrandAnalyzer, LlmSettings
from app.matching_config import DEFAULT_CONFIG
from app.models import (
    BrandProfile,
    ContentAnalysis,
    Judgement,
    KolProfile,
    MatchRequest,
    VideoStats,
    known_content_formats,
)
from app.prompts import MAX_JUDGE_VIDEOS, brand_messages, judge_messages
from app.ranking import rank_candidates
from app.integrations.embeddings import MockEmbedder


def _brand() -> BrandProfile:
    return MockBrandAnalyzer().analyze_brand("website", None)


def _video(number: int, caption: str = "รีวิวแว่น", **fields) -> VideoStats:
    return VideoStats(
        url=f"https://www.tiktok.com/@a/video/{number}",
        views=1000,
        caption=caption,
        **fields,
    )


def _kol(videos: list[VideoStats], handle: str = "a") -> KolProfile:
    return KolProfile(
        handle=handle,
        url=f"https://www.tiktok.com/@{handle}",
        followers=30000,
        videos=videos,
        source="seed",
    )


class _Chat:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        reply = self.replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)


def _analyzer(*replies) -> LiveBrandAnalyzer:
    return LiveBrandAnalyzer(LlmSettings(api_key="k"), chat=_Chat(*replies))


def _judge_reply(**content) -> dict:
    return {
        "creators": [
            {
                "handle": "a",
                "judge": 0.8,
                "audience": 0.5,
                "audience_confidence": 0.5,
                "reason": "เหมาะ",
                **content,
            }
        ]
    }


def _search_item(**fields) -> dict:
    return {
        "id": "1",
        "text": "รีวิวแว่น",
        "authorMeta": {"name": "a", "fans": 10},
        "hashtags": [{"name": "#Glasses"}, {"name": "glasses"}, {"name": " "}, "gaming"],
        "isAd": True,
        **fields,
    }


def test_search_mapper_keeps_hashtags_and_the_ad_label():
    video = map_search_item(_search_item())["videos"][0]

    assert video["hashtags"] == ["Glasses", "gaming"]
    assert video["is_ad"] is True


def test_profile_mapper_keeps_hashtags_and_the_ad_label():
    video = enrich_from_items([_search_item()], followers=10)["videos"][0]

    assert video["hashtags"] == ["Glasses", "gaming"]
    assert video["is_ad"] is True


def test_a_missing_ad_label_is_not_an_ad():
    assert map_search_item(_search_item(isAd=None))["videos"][0]["is_ad"] is False


def test_stored_videos_without_the_new_fields_still_load():
    video = VideoStats.model_validate({"url": None, "caption": "x"})

    assert video.hashtags == []
    assert video.is_ad is False


def test_judge_card_numbers_every_video_up_to_the_limit():
    videos = [_video(number, f"caption {number}") for number in range(1, 13)]

    user = judge_messages(_brand(), [_kol(videos)])[1]["content"]

    assert f"{MAX_JUDGE_VIDEOS}. caption {MAX_JUDGE_VIDEOS}" in user
    assert "caption 11" not in user


def test_judge_card_shows_ad_label_and_only_hashtags_not_in_the_caption():
    video = _video(1, "รีวิวแว่น #Gaming", hashtags=["gaming", "แว่น"], is_ad=True)

    user = judge_messages(_brand(), [_kol([video])])[1]["content"]

    assert "1. [ad] รีวิวแว่น #Gaming #แว่น" in user


def test_judge_numbers_skip_videos_without_captions():
    kol = _kol([_video(1, ""), _video(2, "รีวิว"), _video(3, "สอน")])
    analyzer = _analyzer(
        _judge_reply(style_fit=0.7, style_reason="เข้ากัน", style_evidence=[2])
    )

    content = analyzer.judge(_brand(), [kol])[0].content

    assert content.evidence_urls == ["https://www.tiktok.com/@a/video/3"]


def test_judge_maps_the_content_fields():
    analyzer = _analyzer(
        _judge_reply(
            content_formats=["review", "dance", "REVIEW", "tutorial"],
            content_style="รีวิวจริงจัง",
            style_fit=1.4,
            style_reason="เข้ากัน",
            style_evidence=[1, 1, 99],
        )
    )

    content = analyzer.judge(_brand(), [_kol([_video(1)])])[0].content

    assert content == ContentAnalysis(
        formats=["review", "tutorial"],
        style="รีวิวจริงจัง",
        style_fit=1.0,
        style_reason="เข้ากัน",
        evidence_urls=["https://www.tiktok.com/@a/video/1"],
    )


def test_style_fit_without_a_real_video_as_evidence_is_dropped():
    analyzer = _analyzer(
        _judge_reply(
            content_style="รีวิว",
            style_fit=0.9,
            style_reason="เดา",
            style_evidence=[99],
        )
    )

    content = analyzer.judge(_brand(), [_kol([_video(1)])])[0].content

    assert content.style == "รีวิว"
    assert content.style_fit is None
    assert content.style_reason == ""


def test_malformed_content_fields_keep_the_scores():
    analyzer = _analyzer(
        _judge_reply(
            content_formats="review",
            content_style=7,
            style_fit="high",
            style_reason=None,
            style_evidence="1",
        )
    )

    judgement = analyzer.judge(_brand(), [_kol([_video(1)])])[0]

    assert judgement.judge == 0.8
    assert judgement.content is None


def test_creator_without_captions_has_no_content_analysis():
    analyzer = _analyzer(_judge_reply(style_fit=0.9, style_evidence=[1]))

    assert analyzer.judge(_brand(), [_kol([_video(1, "")])])[0].content is None


def test_known_content_formats_ignores_non_lists():
    assert known_content_formats("review") == []
    assert known_content_formats(None) == []


def test_brand_analysis_keeps_known_formats_only():
    data = _brand().model_dump(mode="json")
    data["content_formats"] = "review, dance, tutorial"
    data["content_style"] = "Hands-on."

    profile = _analyzer(data).analyze_brand("site", None)

    assert profile.content_formats == ["review", "tutorial"]
    assert profile.content_style == "Hands-on."


def test_brief_stored_without_the_new_fields_still_loads():
    data = _brand().model_dump(mode="json")
    for name in ("content_formats", "content_style", "content_style_th"):
        del data[name]

    assert BrandProfile.model_validate(data).content_formats == []


def test_prompts_ask_for_the_formats_and_show_them_to_the_judge():
    assert "content_formats" in brand_messages("site", None)[0]["content"]

    system, user = (m["content"] for m in judge_messages(_brand(), [_kol([_video(1)])]))

    assert '"review"' in system
    assert "content_formats: review, comparison, tutorial" in user
    assert "Hands-on gear reviews" in user


def _ranked(content: ContentAnalysis | None):
    kol = _kol([_video(1)])

    def judge(brief, kols):
        return [
            Judgement(
                handle=kol.handle,
                judge=0.8,
                audience=0.5,
                audience_confidence=0.5,
                reason="เหมาะ",
                content=content,
            )
        ]

    return rank_candidates(
        _brand(),
        [kol],
        embedder=MockEmbedder(),
        judge_fn=judge,
        brand_aliases=[],
        competitor_aliases=[],
        cfg=DEFAULT_CONFIG,
        now=datetime.now(UTC),
    ).ranked[0]


def test_style_fit_does_not_move_any_score():
    low = _ranked(ContentAnalysis(style_fit=0.0))
    high = _ranked(ContentAnalysis(style_fit=1.0))

    assert low.content.style_fit == 0.0
    assert high.content.style_fit == 1.0
    assert low.final == high.final
    assert low.scores == high.scores


def test_demo_graph_returns_content_analysis_for_judged_creators():
    request = MatchRequest(
        website_url="https://brand.example",
        facebook_url="https://facebook.com/brand",
    )
    final = build_graph(replace(mock_integrations())).invoke(initial_state(request))

    assert final["stage"] == "done"
    judged = [row for row in final["ranked"] if row.scores.judged]
    assert any(row.content and row.content.style_fit is not None for row in judged)
    assert all(row.content is None for row in final["ranked"] if not row.scores.judged)
