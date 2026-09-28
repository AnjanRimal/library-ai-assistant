from library_assistant.embeddings import HashingEmbedder, tokenize
from library_assistant.models import Book
from library_assistant.retrieval import VectorIndex


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The Containers and Books for Testing") == ["container", "test"]


def test_embeddings_are_unit_length_and_deterministic():
    embedder = HashingEmbedder()
    first = embedder.embed(["spring boot rest api"])
    second = HashingEmbedder().embed(["spring boot rest api"])
    assert abs(float((first[0] ** 2).sum()) - 1.0) < 1e-5
    assert (first == second).all()


def test_search_ranks_the_most_relevant_book_first(index):
    hits = index.search("infrastructure as code with terraform on aws", k=3)
    assert hits[0].book.id == "b06"
    assert hits[0].score > hits[-1].score


def test_search_respects_k(index):
    assert len(index.search("programming", k=2)) == 2


def test_blank_query_and_empty_index_return_nothing(index):
    assert index.search("   ") == []
    assert VectorIndex(HashingEmbedder()).build([]).search("anything") == []


def test_unrelated_query_scores_below_threshold():
    book = Book(id="x", title="Gardening", authors=["A"], isbn="1", summary="Roses and tulips.")
    idx = VectorIndex(HashingEmbedder()).build([book])
    assert idx.search("kubernetes networking") == []
