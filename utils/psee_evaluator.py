from .box_filtering import filter_boxes
from .coco_eval import evaluate_detection


def evaluate_list(result_boxes_list,
                  gt_boxes_list,
                  height: int,
                  width: int,
                  camera: str = 'gen1',
                  apply_bbox_filters: bool = True,
                  downsampled_by_2: bool = False,
                  return_aps: bool = True):
    assert camera in {'gen1', 'gen4'}

    if camera == 'gen1':
        classes = ("car", "pedestrian")
    elif camera == 'gen4':
        classes = ("pedestrian", "two-wheeler", "car")
    else:
        raise NotImplementedError

    if apply_bbox_filters:
        # Default values taken from: https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/0393adea2bf22d833893c8cb1d986fcbe4e6f82d/src/psee_evaluator.py#L23-L24
        # min_box_diag = 60 if camera == 'gen4' else 30
        min_box_diag = 60 if camera == 'gen4' else 12
        # In the supplementary mat, they say that min_box_side is 20 for gen4.
        # min_box_side = 20 if camera == 'gen4' else 10
        min_box_side = 20 if camera == 'gen4' else 4
        if downsampled_by_2:
            assert min_box_diag % 2 == 0
            min_box_diag //= 2
            assert min_box_side % 2 == 0
            min_box_side //= 2

        # half_sec_us = int(5e5)
        # filter_boxes_fn = lambda x: filter_boxes(x, half_sec_us, min_box_diag, min_box_side)

        # gt_boxes_list = map(filter_boxes_fn, gt_boxes_list)
        # # NOTE: We also filter the prediction to follow the prophesee protocol of evaluation.
        # result_boxes_list = map(filter_boxes_fn, result_boxes_list)

        half_sec_us = int(5e5)
        gt_boxes_list = [filter_boxes(x, half_sec_us, min_box_diag, min_box_side) for x in gt_boxes_list]
        result_boxes_list = [filter_boxes(x, half_sec_us, min_box_diag, min_box_side) for x in result_boxes_list]

    return evaluate_detection(gt_boxes_list, result_boxes_list,
                              height=height, width=width,
                              classes=classes, return_aps=return_aps)


# * ############################################################
# * ############################################################
# * ############################################################

from typing import Any, List, Optional, Dict
from warnings import warn

import numpy as np

# from utils.evaluation.prophesee.evaluation import evaluate_list


class PropheseeEvaluator:
    LABELS = 'lables'
    PREDICTIONS = 'predictions'

    def __init__(self, dataset: str, downsample_by_2: bool):
        super().__init__()
        assert dataset in {'gen1', 'gen4'}
        self.dataset = dataset
        self.downsample_by_2 = downsample_by_2

        self._buffer = None
        self._buffer_empty = True
        self._reset_buffer()

    def _reset_buffer(self):
        self._buffer_empty = True
        self._buffer = {
            self.LABELS: list(),
            self.PREDICTIONS: list(),
        }

    def _add_to_buffer(self, key: str, value: List[np.ndarray]):
        assert isinstance(value, list)
        for entry in value:
            assert isinstance(entry, np.ndarray)
        self._buffer_empty = False
        assert self._buffer is not None
        self._buffer[key].extend(value)

    def _get_from_buffer(self, key: str) -> List[np.ndarray]:
        assert not self._buffer_empty
        assert self._buffer is not None
        return self._buffer[key]

    def add_predictions(self, predictions: List[np.ndarray]):
        self._add_to_buffer(self.PREDICTIONS, predictions)

    def add_labels(self, labels: List[np.ndarray]):
        self._add_to_buffer(self.LABELS, labels)

    def reset_buffer(self) -> None:
        # E.g. call in on_validation_epoch_start
        self._reset_buffer()

    def has_data(self):
        return not self._buffer_empty

    def evaluate_buffer(self, img_height: int, img_width: int) -> Optional[Dict[str, Any]]:
        # e.g call in on_validation_epoch_end
        if self._buffer_empty:
            warn("Attempt to use prophesee evaluation buffer, but it is empty", UserWarning, stacklevel=2)
            return

        labels = self._get_from_buffer(self.LABELS)
        predictions = self._get_from_buffer(self.PREDICTIONS)
        assert len(labels) == len(predictions)
        metrics = evaluate_list(result_boxes_list=predictions,
                                gt_boxes_list=labels,
                                height=img_height,
                                width=img_width,
                                apply_bbox_filters=True,
                                downsampled_by_2=self.downsample_by_2,
                                camera=self.dataset)
        return metrics
