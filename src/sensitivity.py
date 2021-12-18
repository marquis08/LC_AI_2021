import torch
from torch import Tensor
from typing import Any, Callable, Optional
from torchmetrics.functional.classification.stat_scores import _reduce_stat_scores, _stat_scores_update
from torchmetrics.utilities.enums import AverageMethod, MDMCAverageMethod
from torchmetrics.classification.stat_scores import StatScores

def _sensitivity_compute(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    average: str,
    mdmc_average: Optional[str],) -> Tensor:


    numerator = tp
    denominator = tp + fn
    if average == AverageMethod.NONE and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        # a class is not present if there exists no TPs, no FPs, and no FNs
        meaningless_indeces = torch.nonzero((tp | fn | fp) == 0).cpu()
        numerator[meaningless_indeces, ...] = -1
        denominator[meaningless_indeces, ...] = -1
    return _reduce_stat_scores(
        numerator=numerator,
        denominator=denominator,
        weights=None if average != AverageMethod.WEIGHTED else denominator,
        average=average,
        mdmc_average=mdmc_average,
    )

class Sensitivity(StatScores):
    
    is_differentiable = False
    higher_is_better = True

    def __init__(
        self,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        average: str = "micro",
        mdmc_average: Optional[str] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        allowed_average = ["micro", "macro", "weighted", "samples", "none", None]
        if average not in allowed_average:
            raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

        super().__init__(
            reduce="macro" if average in ["weighted", "none", None] else average,
            mdmc_reduce=mdmc_average,
            threshold=threshold,
            top_k=top_k,
            num_classes=num_classes,
            multiclass=multiclass,
            ignore_index=ignore_index,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.average = average

    def compute(self) -> Tensor:
        """Computes the specificity score based on inputs passed in to ``update`` previously.
        Return:
            The shape of the returned tensor depends on the ``average`` parameter
            - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
            - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
              of classes
        """
        tp, fp, tn, fn = self._get_final_stats()
        return _sensitivity_compute(tp, fp, tn, fn, self.average, self.mdmc_reduce)