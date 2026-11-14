from __future__ import annotations

from easyrag.search import BM25, reciprocal_rank_fusion, tokenize


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("The cosine-distance, explained!") == ["cosine", "distance", "explained"]
    # A query made only of stop words keeps its tokens instead of becoming empty.
    assert tokenize("what is it") == ["what", "is", "it"]
    assert tokenize("what is it", keep_stopwords=True) == ["what", "is", "it"]


def test_bm25_ranks_the_matching_document_first():
    documents = [
        "chroma stores vectors and supports cosine distance",
        "a sourdough starter needs flour and water",
        "the cosine distance between two vectors is a similarity measure",
    ]
    ranked = BM25(documents).top("cosine distance", limit=3)
    order = [index for index, _ in ranked]
    assert set(order[:2]) == {0, 2}
    assert 1 not in order
    assert ranked[0][1] > 0


def test_bm25_top_skips_documents_without_matches():
    ranked = BM25(["alpha", "beta", "gamma"]).top("zebra", limit=5)
    assert ranked == []


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [1, 4, 2]])
    assert fused[1] > fused[2] > fused[4] > fused[3]


def test_reciprocal_rank_fusion_supports_weights():
    fused = reciprocal_rank_fusion([[1], [2]], weights=[1.0, 10.0])
    assert fused[2] > fused[1]
