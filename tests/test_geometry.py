import numpy as np
import pytest

from emberwatch.geometry import box_area, box_iou, nms


def test_iou_known_values():
    a = [[0, 0, 2, 2]]
    b = [[1, 1, 3, 3], [0, 0, 2, 2], [5, 5, 6, 6], [0, 0, 1, 2]]
    iou = box_iou(a, b)[0]
    assert iou[0] == pytest.approx(1 / 7)  # inter 1, union 4 + 4 - 1
    assert iou[1] == pytest.approx(1.0)
    assert iou[2] == 0.0
    assert iou[3] == pytest.approx(0.5)


def test_iou_degenerate_box_is_zero():
    assert box_iou([[1, 1, 1, 1]], [[1, 1, 1, 1]])[0, 0] == 0.0
    assert box_area(np.array([[2, 2, 1, 1]]))[0] == 0.0


def test_nms_suppresses_overlap_keeps_best():
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
    assert nms(boxes, [0.6, 0.9, 0.8], 0.5) == [1, 2]


def test_nms_threshold_boundary():
    boxes = [[0, 0, 10, 10], [0, 0, 10, 5]]  # IoU exactly 0.5
    assert nms(boxes, [0.9, 0.8], 0.5) == [0, 1]  # equal to threshold is kept
    assert nms(boxes, [0.9, 0.8], 0.49) == [0]


def test_nms_empty():
    assert nms(np.zeros((0, 4)), np.zeros(0)) == []
