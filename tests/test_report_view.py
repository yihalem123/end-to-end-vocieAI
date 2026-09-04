"""The report template escapes everything it renders."""
from server.postcall.report_view import render_report_html


def test_caller_markup_renders_as_text_not_html() -> None:
    # Autoescaping is the template engine's job now, not hand-placed escape()
    # calls: a hostile or accidental tag in an utterance must never execute.
    html = render_report_html({
        "call_id": "c1", "session_state": "completed", "score": 0.5, "fields": {},
        "conversation": [{"role": "caller", "text": "<script>alert(1)</script> five years",
                          "utterance_id": "c1:u1"}],
        "reasons": ["<b>bold</b>"],
    })
    assert "<script" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt; five years" in html
    assert "&lt;b&gt;bold&lt;/b&gt;" in html


def test_report_stays_javascript_free() -> None:
    html = render_report_html({"call_id": "c2", "session_state": "completed",
                               "score": 0.9, "fields": {}, "conversation": []})
    assert "<script" not in html
    assert 'id="themeToggle"' in html          # the checkbox-driven theme switch
