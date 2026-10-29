"""Offline embedder, cache, and a mocked local model."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace

import pytest

from app.graph import build_graph, initial_state
from app.integrations import build_integrations, mock_integrations
from app.integrations.embeddings import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    EmbedError,
    EmbeddingCache,
    MockEmbedder,
    brand_query_text,
    clamp_unit,
    cosine_similarity,
    document_prefix,
    embedding_cache_key,
    hashed_bow,
    kol_document_text,
    kol_title,
    l2_normalize,
)
from app.integrations.embeddings_live import (
    DEFAULT_EMBED_MODEL,
    EmbedConfigError,
    SentenceTransformerEmbedder,
    clear_sentence_transformer_cache,
)
from app.integrations.llm import MockBrandAnalyzer
from app.models import KolProfile, MatchRequest


@pytest.fixture(autouse=True)
def _reset_loaded_models():
    clear_sentence_transformer_cache()
    yield
    clear_sentence_transformer_cache()


def _request(top_k: int = 10) -> MatchRequest:
    return MatchRequest(
        website_url="https://example-brand.co.th",
        facebook_url="https://www.facebook.com/example-brand",
        top_k=top_k,
    )


def _fixture(handle: str, **overrides) -> dict:
    raw = {
        "handle": handle,
        "url": f"https://www.tiktok.com/@{handle.lstrip('@')}",
        "nickname": handle,
        "followers": 50_000,
        "bio": "Mock Thai gaming creator.",
        "source": "apify",
        "text_language": "th",
        "location_created": "TH",
        "topics": ["gaming", "tech"],
        "query_group": 1,
    }
    raw.update(overrides)
    return raw


def _run(graph, request):
    names: list[str] = []
    final: dict = {}
    for mode, payload in graph.stream(
        initial_state(request),
        stream_mode=["updates", "values"],
    ):
        if mode == "updates":
            names.extend(payload.keys())
        else:
            final = payload
    return names, final


class _DType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


def _fake_torch() -> types.ModuleType:
    module = types.ModuleType("torch")
    module.float32 = _DType("float32")  # type: ignore[attr-defined]
    module.bfloat16 = _DType("bfloat16")  # type: ignore[attr-defined]
    module.float16 = _DType("float16")  # type: ignore[attr-defined]
    return module


class _NamedModel:
    """Stand-in with embeddinggemma-2 prompt names."""

    def __init__(self) -> None:
        self.prompts = {
            "SearchQuery": QUERY_PREFIX,
            "Document": DOCUMENT_PREFIX,
        }
        self.calls: list[dict] = []
        self.max_seq_length = 512

    def encode(self, texts, prompt_name=None, normalize_embeddings=False, truncate_dim=None, **kwargs):
        self.calls.append(
            {
                "texts": list(texts),
                "prompt_name": prompt_name,
                "normalize_embeddings": normalize_embeddings,
                "truncate_dim": truncate_dim,
            }
        )
        return [[1.0, 0.0, 0.0] if index == 0 else [0.0, 1.0, 0.0] for index, _ in enumerate(texts)]


class _LoadingModel:
    instances: list[_LoadingModel] = []

    def __init__(self, model_name, device=None, config_kwargs=None, model_kwargs=None):
        self.model_name = model_name
        self.device = device
        self.config_kwargs = config_kwargs
        self.model_kwargs = model_kwargs
        self.max_seq_length = 512
        self.prompts = {
            "SearchQuery": QUERY_PREFIX,
            "Document": DOCUMENT_PREFIX,
        }
        self.calls: list[dict] = []
        type(self).instances.append(self)

    def encode(self, texts, prompt_name=None, normalize_embeddings=False, truncate_dim=None, **kwargs):
        self.calls.append(
            {
                "texts": list(texts),
                "prompt_name": prompt_name,
                "normalize_embeddings": normalize_embeddings,
                "truncate_dim": truncate_dim,
            }
        )
        return [[1.0, 0.0] for _ in texts]


def _install_loader(monkeypatch, cls=_LoadingModel):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    _LoadingModel.instances = []


def _block_sentence_transformers(monkeypatch):
    real_import = __import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocked)


def test_mock_embedder_is_deterministic_and_unit_length():
    embedder = MockEmbedder()
    first = embedder.similarity("blue light glasses", ["gaming glasses", "สูตรอาหาร"])
    second = embedder.similarity("blue light glasses", ["gaming glasses", "สูตรอาหาร"])
    assert first == second
    assert embedder.similarity("q", []) == []
    vector = hashed_bow(QUERY_PREFIX + "แว่นกันแสงสีฟ้า gaming", 256)
    assert len(vector) == 256
    assert sum(value * value for value in vector) == pytest.approx(1.0)
    assert hashed_bow(QUERY_PREFIX + "glasses") != hashed_bow(DOCUMENT_PREFIX + "glasses")
    assert 0.0 <= first[0] <= 1.0
    assert 0.0 <= first[1] <= 1.0


def test_cosine_ranks_a_closer_document_first():
    scores = MockEmbedder().similarity(
        "blue light glasses gaming eyewear esports",
        [
            "blue light glasses gaming eyewear esports review",
            "สูตรต้มยำอาหารเย็นและขนมหวาน",
        ],
    )
    assert scores[0] > scores[1]

    left = l2_normalize([3.0, 0.0])
    right = l2_normalize([0.0, 4.0])
    same = l2_normalize([3.0, 0.0])
    assert cosine_similarity(left, same) == pytest.approx(1.0)
    assert cosine_similarity(left, right) == pytest.approx(0.0)
    assert cosine_similarity(left, [0.0, 0.0]) == 0.0
    assert clamp_unit(-0.2) == 0.0
    assert clamp_unit(1.4) == 1.0


def test_brand_and_kol_text_include_the_retrieval_fields():
    brand = MockBrandAnalyzer().analyze_brand("website", None)
    text = brand_query_text(brand)
    assert brand.category in text
    assert brand.key_products[0] in text
    assert brand.target_audience.interests[0] in text
    assert brand.topic_keywords_en[0] in text
    assert brand.competitors[0] in text
    assert "audience:" in text

    kol = KolProfile(
        handle="Alpha",
        url="https://www.tiktok.com/@Alpha",
        nickname="พลอย",
        bio="รีวิวแว่น #GamingGlasses",
        content_summary="Recent posts: desk setup | Hashtags: #eyewear",
        source="seed",
    )
    document = kol_document_text(kol)
    assert "พลอย" in document
    assert "รีวิวแว่น" in document
    assert "desk setup" in document
    assert "#GamingGlasses" in document
    assert "#eyewear" in document
    assert kol_title(kol) == "พลอย"
    assert document_prefix("พลอย") == "title: พลอย | text: "
    assert document_prefix(None) == DOCUMENT_PREFIX
    untitled = KolProfile(
        handle="Bare",
        url="https://www.tiktok.com/@Bare",
        source="seed",
    )
    assert kol_title(untitled) == "Bare"


def test_cache_hit_miss_and_atomic_write(tmp_path):
    cache = EmbeddingCache(tmp_path)
    assert cache.get("embeddinggemma-2", QUERY_PREFIX, "hello", 768) is None
    cache.put("embeddinggemma-2", QUERY_PREFIX, "hello", 768, [1.0, 0.0])
    assert cache.get("embeddinggemma-2", QUERY_PREFIX, "hello", 768) == [1.0, 0.0]
    assert cache.get("embeddinggemma-2", QUERY_PREFIX, "other", 768) is None
    assert cache.get("other-model", QUERY_PREFIX, "hello", 768) is None
    assert cache.get("embeddinggemma-2", DOCUMENT_PREFIX, "hello", 768) is None
    assert cache.get("embeddinggemma-2", QUERY_PREFIX, "hello", 256) is None

    path = cache.path_for("embeddinggemma-2", QUERY_PREFIX, "hello", 768)
    assert path.name == embedding_cache_key("embeddinggemma-2", QUERY_PREFIX, "hello", 768) + ".json"
    assert embedding_cache_key("m", "p", "t", 768) != embedding_cache_key("m", "p", "t", 256)
    assert list(tmp_path.glob("*.tmp")) == []

    path.write_text("{", encoding="utf-8")
    assert cache.get("embeddinggemma-2", QUERY_PREFIX, "hello", 768) is None

    path.write_text(
        json.dumps(
            {
                "model": "nope",
                "prefix": QUERY_PREFIX,
                "text": "hello",
                "dim": 768,
                "vector": [1.0],
            }
        ),
        encoding="utf-8",
    )
    assert cache.get("embeddinggemma-2", QUERY_PREFIX, "hello", 768) is None


def test_first_pass_similarity_orders_creators():
    fixtures = [
        _fixture(
            "aaa_far",
            nickname="อาหาร",
            bio="สูตรอาหารต้มยำและขนมหวาน",
            topics=["cooking", "food"],
        ),
        _fixture(
            "zzz_near",
            nickname="gaming glasses",
            bio="blue light glasses gaming eyewear esports",
            topics=["gaming", "eyewear", "blue light glasses"],
        ),
    ]
    final = build_graph(
        mock_integrations(fixtures),
        min_candidates=1,
    ).invoke(initial_state(_request(top_k=2)))

    assert final["stage"] == "done"
    by_handle = {item.kol.handle: item for item in final["ranked"]}
    assert by_handle["zzz_near"].scores.sim > by_handle["aaa_far"].scores.sim
    assert final["ranked"][0].kol.handle == "zzz_near"
    for item in final["ranked"]:
        assert item.final == pytest.approx(item.scores.F * item.quality)


def test_top_k_above_enrich_cap_is_not_capped():
    fixtures = [
        _fixture(f"kol{index:02d}", followers=1_000 + index)
        for index in range(40)
    ]
    request = _request(top_k=30)
    final = build_graph(
        mock_integrations(fixtures),
        min_candidates=1,
    ).invoke(initial_state(request))

    assert final["stage"] == "done"
    assert len(final["ranked"]) == 40
    assert len(final["results"]) == 30


def test_enrich_limit_must_be_positive():
    with pytest.raises(ValueError, match="enrich_limit"):
        build_graph(enrich_limit=0)


def test_prompt_name_encode_normalizes_and_truncates(tmp_path):
    model = _NamedModel()
    embedder = SentenceTransformerEmbedder(model=model, cache_dir=tmp_path, dim=768, batch_size=2)
    scores = embedder.similarity("glasses", ["glasses", "pasta", "other"], titles=["A", "B", "C"])

    assert [call["prompt_name"] for call in model.calls] == ["SearchQuery", "Document", "Document"]
    assert model.calls[0]["texts"] == ["glasses"]
    assert model.calls[1]["texts"] == ["glasses", "pasta"]
    assert model.calls[2]["texts"] == ["other"]
    assert all(call["normalize_embeddings"] is True for call in model.calls)
    assert all(call["truncate_dim"] == 768 for call in model.calls)
    assert all(not text.startswith("task:") for text in model.calls[0]["texts"])
    assert all(not text.startswith("title:") for call in model.calls[1:] for text in call["texts"])
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)

    calls_after_first = len(model.calls)
    again = embedder.similarity("glasses", ["glasses", "pasta", "other"], titles=["A", "B", "C"])
    assert again == scores
    assert len(model.calls) == calls_after_first


def test_cache_miss_when_dimension_changes(tmp_path):
    model = _NamedModel()
    wide = SentenceTransformerEmbedder(model=model, cache_dir=tmp_path, dim=768)
    wide.similarity("q", ["doc"])
    narrow = SentenceTransformerEmbedder(model=model, cache_dir=tmp_path, dim=256)
    narrow.similarity("q", ["doc"])
    assert [call["truncate_dim"] for call in model.calls] == [768, 768, 256, 256]


def test_manual_document_prompt_uses_nickname_or_handle(tmp_path):
    class Manual:
        def __init__(self) -> None:
            self.seen: list[tuple] = []
            self.prompts: dict = {}

        def encode(self, texts, prompt_name=None, **kwargs):
            self.seen.append((prompt_name, list(texts), kwargs.get("truncate_dim")))
            return [[1.0, 0.0] for _ in texts]

    model = Manual()
    embedder = SentenceTransformerEmbedder(model=model, cache_dir=tmp_path, dim=128)
    embedder.similarity("glasses", ["bio text"], titles=["น้องพี่เกม"])
    assert model.seen[0] == (None, [QUERY_PREFIX + "glasses"], 128)
    assert model.seen[1] == (None, ["title: น้องพี่เกม | text: bio text"], 128)

    embedder.similarity("glasses", ["bio text"], titles=[None])
    assert model.seen[-1][1] == [DOCUMENT_PREFIX + "bio text"]


def test_model_is_loaded_once_text_only_in_float32(monkeypatch, tmp_path):
    _install_loader(monkeypatch)
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.delenv("EMBED_DEVICE", raising=False)
    monkeypatch.delenv("EMBED_DIM", raising=False)
    monkeypatch.delenv("EMBED_DTYPE", raising=False)

    first = SentenceTransformerEmbedder(cache_dir=tmp_path, dim=768)
    second = SentenceTransformerEmbedder(cache_dir=tmp_path, dim=256)
    first.similarity("q", ["d"])
    second.similarity("q2", ["d2"])

    assert len(_LoadingModel.instances) == 1
    loaded = _LoadingModel.instances[0]
    assert loaded.model_name == DEFAULT_EMBED_MODEL
    assert loaded.device == "cpu"
    assert loaded.config_kwargs == {"vision_config": None, "audio_config": None}
    assert loaded.model_kwargs["dtype"].name == "float32"
    assert loaded.model_kwargs["dtype"].name != "float16"
    assert loaded.max_seq_length == 8192
    assert [call["truncate_dim"] for call in loaded.calls] == [768, 768, 256, 256]


def test_bfloat16_is_allowed_and_float16_is_rejected(monkeypatch, tmp_path):
    _install_loader(monkeypatch)
    monkeypatch.setenv("EMBED_DTYPE", "bfloat16")
    embedder = SentenceTransformerEmbedder.from_env(cache_dir=tmp_path)
    embedder.similarity("q", ["d"])
    assert _LoadingModel.instances[0].model_kwargs["dtype"].name == "bfloat16"

    monkeypatch.setenv("EMBED_DTYPE", "float16")
    with pytest.raises(EmbedConfigError, match="float16"):
        SentenceTransformerEmbedder.from_env()
    monkeypatch.setenv("EMBED_DIM", "300")
    with pytest.raises(EmbedConfigError, match="EMBED_DIM"):
        SentenceTransformerEmbedder.from_env()


def test_missing_extra_and_load_failure_are_clear(monkeypatch, tmp_path):
    _block_sentence_transformers(monkeypatch)
    embedder = SentenceTransformerEmbedder(cache_dir=tmp_path)
    with pytest.raises(EmbedError, match="uv sync --extra embed") as missing:
        embedder.similarity("q", ["d"])
    assert "gated" not in str(missing.value).lower()

    class Boom:
        def __init__(self, *args, **kwargs):
            raise OSError("401 from secret-token download")

    monkeypatch.undo()
    _install_loader(monkeypatch, Boom)
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    failed = SentenceTransformerEmbedder(cache_dir=tmp_path)
    with pytest.raises(EmbedError, match="Could not load embedding model") as exc:
        failed.similarity("q", ["d"])
    message = str(exc.value)
    assert "secret-token" not in message
    assert "[redacted]" in message
    assert "gated" not in message.lower()
    assert "license" not in message.lower()


def test_unreachable_embedder_stops_before_judging(monkeypatch, tmp_path):
    _block_sentence_transformers(monkeypatch)
    integrations = replace(
        mock_integrations(),
        embedder=SentenceTransformerEmbedder(cache_dir=tmp_path),
    )
    names, final = _run(
        build_graph(integrations, min_candidates=1),
        _request(),
    )

    assert final["stage"] == "failed"
    assert any(
        error.startswith("first_pass_rank:") and "sentence-transformers" in error
        for error in final["errors"]
    )
    assert "enrich_kols" not in names
    assert "judge_and_rank" not in names
    assert "rank_and_output" not in names


def test_build_integrations_live_embed_does_not_load_the_model(monkeypatch):
    for name in ("EMBED_MODEL", "EMBED_DEVICE", "EMBED_DIM", "EMBED_DTYPE", "EMBED_BATCH_SIZE"):
        monkeypatch.delenv(name, raising=False)
    before = "sentence_transformers" in sys.modules
    bundle = build_integrations(live_embed=True)
    try:
        assert isinstance(bundle.embedder, SentenceTransformerEmbedder)
        assert isinstance(bundle.llm, MockBrandAnalyzer)
        assert bundle.embedder.model_name == "google/embeddinggemma-2"
        assert bundle.embedder.device == "cpu"
        assert bundle.embedder.dim == 768
        assert bundle.embedder.dtype == "float32"
        assert bundle.embedder._model is None
        assert ("sentence_transformers" in sys.modules) is before
        assert "MOCK" not in (bundle.embedder.__class__.__doc__ or "")
    finally:
        clear_sentence_transformer_cache()

    assert isinstance(build_integrations().embedder, MockEmbedder)
