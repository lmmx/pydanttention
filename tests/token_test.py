from pydanttention.main import Token


def test_token_a():
    tok = Token(tok=0, vocab=list("ab"))
    assert repr(tok) == "Token(tok=0, char='a')"


def test_token_b():
    tok = Token(tok=1, vocab=list("ab"))
    assert repr(tok) == "Token(tok=1, char='b')"
