from pydanttention.main import ManualTransformer


def test_original_implementation():
    original_run = ManualTransformer()
    assert original_run.correct == original_run.total == 27
