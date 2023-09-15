from pydanttention.original.vogel import VogelManualTransformer


def test_original_implementation():
    original_run = VogelManualTransformer()
    assert original_run.correct == original_run.total == 27
