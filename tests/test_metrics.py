import pytest

from emberwatch.metrics import average_precision, evaluate_class, operating_point, pr_curve

GT = [("fire", [0, 0, 10, 10]), ("fire", [20, 20, 30, 30])]


def test_ap_toy_example_known_answer():
    # dets by score: TP (.9), FP (.8), TP (.7) -> recall .5/.5/1, precision 1/.5/.667
    preds = [
        [
            ("fire", 0.9, [0, 0, 10, 10]),
            ("fire", 0.8, [100, 100, 110, 110]),
            ("fire", 0.7, [20, 20, 30, 30]),
        ]
    ]
    r = evaluate_class(preds, [GT], "fire")
    assert r["ap"] == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))


def test_ap_perfect_and_empty():
    perfect = [[("fire", 0.9, b) for _, b in GT]]
    assert evaluate_class(perfect, [GT], "fire")["ap"] == pytest.approx(1.0)
    assert evaluate_class([[]], [GT], "fire")["ap"] == 0.0


def test_duplicate_detection_is_false_positive():
    preds = [[("fire", 0.9, [0, 0, 10, 10]), ("fire", 0.8, [0, 0, 10, 10])]]
    op = operating_point(preds, [GT], "fire", 0.5)
    assert (op["tp"], op["fp"], op["n_gt"]) == (1, 1, 2)
    assert op["precision"] == 0.5 and op["recall"] == 0.5


def test_class_mismatch_not_matched():
    preds = [[("smoke", 0.9, [0, 0, 10, 10])]]
    assert evaluate_class(preds, [GT], "fire")["ap"] == 0.0


def test_score_threshold_changes_operating_point():
    preds = [[("fire", 0.9, [0, 0, 10, 10]), ("fire", 0.3, [20, 20, 30, 30])]]
    assert operating_point(preds, [GT], "fire", 0.5)["recall"] == 0.5
    assert operating_point(preds, [GT], "fire", 0.2)["recall"] == 1.0


def test_envelope_monotone():
    prec, rec, _ = pr_curve([0.9, 0.8, 0.7, 0.6], [True, False, True, True], 3)
    assert average_precision(prec, rec) == pytest.approx(
        1 / 3 * (1.0 + 0.75 + 0.75)
    )  # envelope lifts 0.667 to 0.75
