import importlib.util, pathlib

spec = importlib.util.spec_from_file_location(
    "gen", pathlib.Path(__file__).parent / "generate-dataset.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def test_ratio_one_holds_negatives_equal_to_positives():
    assert gen.negative_target(300, 1.0) == 300
    assert gen.negative_target(130, 1.0) == 130


def test_floor_only_binds_for_a_tiny_corpus():
    # The floor exists for near-empty corpora, not to reshape a real one.
    assert gen.negative_target(10, 1.0) == gen.NEG_FLOOR


def test_ratio_scales_negatives():
    assert gen.negative_target(130, 2.0) == 260
    assert gen.negative_target(200, 0.5) == 100


def test_floor_boundary_steps_at_fifty_positives():
    # Pin both sides of the < 50 threshold: the floor binds at 49
    # positives and the ratio binds at 50. This is a deliberate step,
    # not a bug — see the negative_target docstring.
    assert gen.negative_target(49, 1.0) == 250
    assert gen.negative_target(50, 1.0) == 50
