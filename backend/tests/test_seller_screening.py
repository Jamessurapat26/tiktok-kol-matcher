"""A Search must not recommend seller accounts, and must survive finding only them."""

from __future__ import annotations

from dataclasses import replace

from app.filters import MEDIA_ACCOUNT, SELLER_ACCOUNT
from app.graph import build_graph, initial_state
from app.integrations import mock_integrations
from app.integrations.llm import MockBrandAnalyzer
from app.models import MatchRequest
from app.nodes import make_nodes
from app.prompts import BRAND_ANALYSIS_SYSTEM, JUDGE_SYSTEM
from app.state import MatchState


def _request(top_k: int = 5) -> MatchRequest:
    return MatchRequest(
        website_url="https://example-brand.co.th",
        facebook_url="https://www.facebook.com/examplebrand",
        top_k=top_k,
    )


def _fixture(handle: str, **overrides) -> dict:
    return {
        "handle": handle,
        "url": f"https://www.tiktok.com/@{handle}",
        "nickname": handle,
        "followers": 50_000,
        "bio": "รีวิวเกมและแว่นกันแสง",
        "source": "apify",
        "text_language": "th",
        "location_created": "TH",
        "topics": ["gaming", "tech"],
        "query_group": 1,
        **overrides,
    }


class _TypingJudge(MockBrandAnalyzer):
    def __init__(self, types: dict[str, str], skip: tuple[str, ...] = ()) -> None:
        self.types = types
        self.skip = skip
        self.sent: list[str] = []

    def judge(self, brief, kols):
        self.sent = [kol.handle for kol in kols]
        judgements = [
            item for item in super().judge(brief, kols) if item.handle not in self.skip
        ]
        return [
            item.model_copy(
                update={"account_type": self.types.get(item.handle, "kol")}
            )
            for item in judgements
        ]


def _run(fixtures, llm=None, top_k: int = 5, **cfg):
    integrations = mock_integrations(fixtures)
    if llm is not None:
        integrations = replace(integrations, llm=llm)
    return build_graph(integrations, min_candidates=1, **cfg).invoke(
        initial_state(_request(top_k))
    )


def _handles(final) -> set[str]:
    return {item.kol.handle for item in final["results"]}


def test_accounts_the_judge_types_are_removed_and_counted():
    fixtures = [_fixture(name) for name in ("creator1", "creator2", "farmer", "paper")]
    final = _run(fixtures, _TypingJudge({"farmer": "seller", "paper": "media"}))

    assert final["stage"] == "done"
    assert _handles(final) == {"creator1", "creator2"}
    assert final["rejected"][SELLER_ACCOUNT] == 1
    assert final["rejected"][MEDIA_ACCOUNT] == 1


def test_finding_only_sellers_is_a_finished_search_with_no_results():
    fixtures = [
        _fixture("dealer1", bio="ตัวแทนจำหน่ายแว่น"),
        _fixture("dealer2", bio="ตัวแทนจำหน่ายแว่น"),
    ]
    final = _run(fixtures)

    assert final["stage"] == "done"
    assert final["results"] == []
    assert final["rejected"][SELLER_ACCOUNT] == 2


def test_judge_typing_every_account_as_seller_is_a_finished_search():
    final = _run([_fixture("farmer")], _TypingJudge({"farmer": "seller"}))

    assert final["stage"] == "done"
    assert final["results"] == []


def test_an_account_the_judge_skipped_stays_with_a_warning_flag():
    final = _run(
        [_fixture("creator1"), _fixture("creator2")],
        _TypingJudge({}, skip=("creator2",)),
    )

    flags = {item.kol.handle: item.flags for item in final["results"]}
    assert "unreviewed" in flags["creator2"]
    assert "unreviewed" not in flags["creator1"]


def test_judge_failure_keeps_results_with_a_warning_flag():
    class _BoomJudge(MockBrandAnalyzer):
        def judge(self, brief, kols):
            raise RuntimeError("judge down")

    final = _run([_fixture("creator1")], _BoomJudge())

    assert final["stage"] == "done"
    assert "unreviewed" in final["results"][0].flags


def test_accounts_outside_the_judge_window_are_not_recommended():
    fixtures = [_fixture(f"creator{index:02d}") for index in range(25)]
    llm = _TypingJudge({})
    final = _run(fixtures, llm, top_k=25)
    assert len(llm.sent) == 25
    assert len(final["results"]) == 25

    llm = _TypingJudge({})
    final = _run(fixtures, llm, top_k=5)
    assert len(llm.sent) == 20
    assert all(item.scores.judged for item in final["results"])


def test_plain_sellers_do_not_count_toward_the_search_target():
    brand = MockBrandAnalyzer().analyze_brand("site", None)
    dealer = make_kol("dealer", "ตัวแทนจำหน่ายแว่น")
    creator = make_kol("creator", "รีวิวเกม")
    nodes = make_nodes(mock_integrations(), min_candidates=2, max_search_rounds=3)

    def route(*kols):
        return nodes["route_after_discover"](
            MatchState(
                request=_request(),
                brand_profile=brand,
                discovered=list(kols),
                search_rounds=1,
                stage="discovering_kols",
            )
        )

    assert route(dealer, creator) == "generate_queries"
    assert route(creator, make_kol("creator2", "รีวิวเกม")) == "hard_filter"


def make_kol(handle: str, bio: str):
    from app.models import KolProfile

    return KolProfile(
        handle=handle,
        url=f"https://www.tiktok.com/@{handle}",
        nickname=handle,
        followers=50_000,
        bio=bio,
        language="th",
        region="TH",
        source="apify",
    )


def test_judge_prompt_asks_for_the_account_type():
    assert "account_type" in JUDGE_SYSTEM
    assert "bio offers a way for customers to buy or hire" in " ".join(JUDGE_SYSTEM.split())
    assert "stays \"kol\" even if they also sell courses" in " ".join(JUDGE_SYSTEM.split())


def test_large_thai_brands_have_one_tier_rule():
    prompt = " ".join(BRAND_ANALYSIS_SYSTEM.split())

    assert "Nationwide Thai brand, however large" in prompt
    assert "very large" not in prompt
