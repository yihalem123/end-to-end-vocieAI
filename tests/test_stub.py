"""Stub engine: canned screening replies, pre-split into sentences."""
from server.engine.stub import StubEngine


def test_replies_cycle_in_order() -> None:
    engine = StubEngine()
    first = engine.reply("hello")
    second = engine.reply("I'm a nurse")
    assert first != second


def test_reply_is_a_list_of_sentences() -> None:
    # The speak task opens one TTS stream per sentence and records a mark at
    # each boundary — replies must arrive pre-split so marks are exact.
    engine = StubEngine()
    for _ in range(6):
        sentences = engine.reply("anything")
        assert isinstance(sentences, list)
        assert all(isinstance(s, str) and s.strip() for s in sentences)


def test_engine_never_runs_out() -> None:
    engine = StubEngine()
    replies = [tuple(engine.reply(str(i))) for i in range(20)]
    assert all(replies)  # keeps producing after the script is exhausted
