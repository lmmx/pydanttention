from pydanttention.main import Transformer


def test_implementation():
    model = Transformer()
    model.run()
    expected_log = "a (0): next=b (1) probs=[0. 1.] logits=[1.000e+00 1.024e+03]"
    assert model.logs[0] == expected_log
    assert model.correct == model.total == 27
