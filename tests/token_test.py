from pydanttention.main import Token


def test_token_a():
    tok = Token(idx=0, vocab=list("ab"))
    assert str(tok) == "a (0)"
    assert repr(tok) == "Token(idx=0, char='a')"


def test_token_b():
    tok = Token(idx=1, vocab=list("ab"))
    assert str(tok) == "b (1)"
    assert repr(tok) == "Token(idx=1, char='b')"
