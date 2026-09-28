from library_assistant.tools import TOOL_DEFINITIONS, ToolExecutor


def test_tool_definitions_have_unique_names_and_schemas():
    names = [t["name"] for t in TOOL_DEFINITIONS]
    assert len(names) == len(set(names))
    assert all(t["input_schema"]["type"] == "object" for t in TOOL_DEFINITIONS)


def test_search_returns_ids_and_tracks_seen_books(catalog, index):
    executor = ToolExecutor(catalog, index)
    content, is_error = executor.run("search_catalog", {"query": "spring boot java", "k": 2})
    assert not is_error
    assert "[b05]" in content
    assert "b05" in executor.seen_book_ids
    assert executor.records[0].name == "search_catalog"


def test_search_clamps_k(catalog, index):
    executor = ToolExecutor(catalog, index)
    content, _ = executor.run("search_catalog", {"query": "programming", "k": 500})
    assert content.count("\n") < 10


def test_get_unknown_book_is_a_recoverable_error(catalog, index):
    executor = ToolExecutor(catalog, index)
    content, is_error = executor.run("get_book", {"book_id": "nope"})
    assert is_error and "No book" in content
    assert executor.records[0].is_error


def test_borrow_requires_signed_in_member(catalog, index):
    content, is_error = ToolExecutor(catalog, index).run("borrow_book", {"book_id": "b01"})
    assert is_error and "member id" in content


def test_borrow_decrements_copies(catalog, index):
    before = catalog.get_book("b01").copies
    content, is_error = ToolExecutor(catalog, index, member_id=7).run("borrow_book", {"book_id": "b01"})
    assert not is_error and "Member 7" in content
    assert catalog.get_book("b01").copies == before - 1


def test_borrow_fails_when_no_copies_left(catalog, index):
    content, is_error = ToolExecutor(catalog, index, member_id=7).run("borrow_book", {"book_id": "b16"})
    assert is_error and "no copies" in content


def test_unknown_tool_and_bad_arguments_are_errors(catalog, index):
    executor = ToolExecutor(catalog, index)
    assert executor.run("delete_everything", {})[1]
    content, is_error = executor.run("get_book", {"wrong_arg": 1})
    assert is_error and "Invalid arguments" in content
