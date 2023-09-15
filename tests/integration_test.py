from pydanttention.main import ManualTransformer


def test_original_implementation():
    model = ManualTransformer()
    model.run()
    assert model.correct == model.total == 27
