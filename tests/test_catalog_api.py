"""Tests for the adapter that talks to the Spring Boot library-api."""

import httpx
import pytest

from library_assistant.catalog import CatalogError, LibraryApiCatalog, build_catalog


def _catalog(handler) -> LibraryApiCatalog:
    client = httpx.Client(base_url="http://library-api", transport=httpx.MockTransport(handler))
    return LibraryApiCatalog("http://library-api", client=client)


def test_maps_book_dtos():
    def handler(request):
        return httpx.Response(
            200,
            json=[
                {
                    "id": 1,
                    "title": "Clean Code",
                    "isbn": "978",
                    "available": False,
                    "authorNames": ["Robert Martin"],
                }
            ],
        )

    [book] = _catalog(handler).all_books()
    assert (book.id, book.authors, book.available) == ("1", ["Robert Martin"], False)


def test_missing_book_raises_catalog_error():
    # library-api returns 200 with an empty body when a book isn't found.
    with pytest.raises(CatalogError):
        _catalog(lambda r: httpx.Response(200, content=b"")).get_book("9")


def test_borrow_success_and_failure_messages():
    ok = _catalog(lambda r: httpx.Response(200, text="Book borrowed successfully"))
    assert ok.borrow(1, "1") == "Book borrowed successfully"

    unavailable = _catalog(lambda r: httpx.Response(200, text="Book is not available"))
    with pytest.raises(CatalogError, match="not available"):
        unavailable.borrow(1, "1")


def test_borrow_calls_the_right_endpoint():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        return httpx.Response(200, text="Book borrowed successfully")

    _catalog(handler).borrow(5, "3")
    assert seen["path"] == "/members/5/borrow/3"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        build_catalog("postgres", "")
