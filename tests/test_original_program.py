from pydanttention.vogel_manual_transformer import ManualTransformer


def test_original():
    original_run = ManualTransformer()
    assert original_run.correct == original_run.total == 27
