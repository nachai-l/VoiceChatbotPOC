import pytest

from app.live.transcript_handler import TranscriptHandler


@pytest.fixture
def handler():
    return TranscriptHandler()


class TestAddTurns:
    def test_add_user_appends_user_role(self, handler):
        handler.add_user("Hello")
        assert handler.get_history() == [{"role": "user", "content": "Hello"}]

    def test_add_assistant_appends_assistant_role(self, handler):
        handler.add_assistant("Hi there")
        assert handler.get_history() == [{"role": "assistant", "content": "Hi there"}]

    def test_turns_are_ordered(self, handler):
        handler.add_user("first")
        handler.add_assistant("second")
        handler.add_user("third")
        roles = [m["role"] for m in handler.get_history()]
        assert roles == ["user", "assistant", "user"]

    def test_empty_user_text_falls_back_to_placeholder(self, handler):
        handler.add_user("")
        assert handler.get_history()[0]["content"] == "[audio]"

    def test_none_user_text_falls_back_to_placeholder(self, handler):
        handler.add_user(None)
        assert handler.get_history()[0]["content"] == "[audio]"

    def test_empty_assistant_text_falls_back_to_placeholder(self, handler):
        handler.add_assistant("")
        assert handler.get_history()[0]["content"] == "[audio response]"

    def test_none_assistant_text_falls_back_to_placeholder(self, handler):
        handler.add_assistant(None)
        assert handler.get_history()[0]["content"] == "[audio response]"


class TestGetHistory:
    def test_empty_on_init(self, handler):
        assert handler.get_history() == []

    def test_returns_copy_not_reference(self, handler):
        handler.add_user("hello")
        h = handler.get_history()
        h.append({"role": "user", "content": "injected"})
        assert len(handler.get_history()) == 1  # internal state unchanged


class TestClear:
    def test_clear_empties_history(self, handler):
        handler.add_user("hello")
        handler.add_assistant("hi")
        handler.clear()
        assert handler.get_history() == []

    def test_can_add_after_clear(self, handler):
        handler.add_user("before")
        handler.clear()
        handler.add_user("after")
        assert len(handler.get_history()) == 1
        assert handler.get_history()[0]["content"] == "after"
