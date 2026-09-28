"""Catalog backends.

`LocalCatalog` serves the bundled JSON data and keeps loans in memory.
`LibraryApiCatalog` talks to the Spring Boot library-api
(https://github.com/AnjanRimal/library-api), so the assistant can act on real data.
"""

from __future__ import annotations

import json
import threading
from importlib import resources
from typing import Protocol

import httpx

from .models import Book


class CatalogError(Exception):
    """Raised for expected failures, such as an unknown book or no copies left."""


class Catalog(Protocol):
    def all_books(self) -> list[Book]: ...
    def get_book(self, book_id: str) -> Book: ...
    def borrow(self, member_id: int, book_id: str) -> str: ...


class LocalCatalog:
    def __init__(self, books: list[Book] | None = None):
        if books is None:
            raw = resources.files("library_assistant.data").joinpath("books.json").read_text()
            books = [Book(**b) for b in json.loads(raw)]
        self._books = {b.id: b for b in books}
        self._lock = threading.Lock()

    def all_books(self) -> list[Book]:
        return list(self._books.values())

    def get_book(self, book_id: str) -> Book:
        try:
            return self._books[book_id]
        except KeyError:
            raise CatalogError(f"No book with id '{book_id}'.") from None

    def borrow(self, member_id: int, book_id: str) -> str:
        with self._lock:
            book = self.get_book(book_id)
            if book.copies <= 0:
                raise CatalogError(f"'{book.title}' has no copies available.")
            self._books[book_id] = book.model_copy(update={"copies": book.copies - 1})
        return f"Member {member_id} borrowed '{book.title}'."


class LibraryApiCatalog:
    """Adapter for the Spring Boot library-api REST endpoints."""

    def __init__(self, base_url: str, client: httpx.Client | None = None):
        self._client = client or httpx.Client(base_url=base_url, timeout=10.0)

    @staticmethod
    def _to_book(raw: dict) -> Book:
        authors = raw.get("authors") or raw.get("authorNames") or []
        authors = [a["name"] if isinstance(a, dict) else str(a) for a in authors]
        return Book(
            id=str(raw["id"]),
            title=raw.get("title", ""),
            authors=authors,
            isbn=raw.get("isbn", ""),
            copies=1 if raw.get("available", True) else 0,
        )

    def all_books(self) -> list[Book]:
        resp = self._client.get("/books")
        resp.raise_for_status()
        return [self._to_book(b) for b in resp.json()]

    def get_book(self, book_id: str) -> Book:
        resp = self._client.get(f"/books/{book_id}")
        if resp.status_code == 404 or not resp.content:
            raise CatalogError(f"No book with id '{book_id}'.")
        resp.raise_for_status()
        return self._to_book(resp.json())

    def borrow(self, member_id: int, book_id: str) -> str:
        resp = self._client.post(f"/members/{member_id}/borrow/{book_id}")
        # library-api answers 200 with a message string, even when the borrow fails.
        if resp.status_code >= 400 or "successfully" not in resp.text:
            raise CatalogError(resp.text or f"Borrow failed with HTTP {resp.status_code}.")
        return resp.text


def build_catalog(backend: str, library_api_url: str) -> Catalog:
    if backend == "api":
        return LibraryApiCatalog(library_api_url)
    if backend == "local":
        return LocalCatalog()
    raise ValueError(f"Unknown catalog backend '{backend}'. Use 'local' or 'api'.")
