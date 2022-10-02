import torch
from torchmetrics import Metric
import torchmetrics.utilities.data as tmutils
from torchmetrics.functional import auc as tm_compute_auc
from torchmetrics.utilities.imports import _TORCH_LOWER_1_6

from typing import Union, Optional, Any, List, Tuple


class BinnedROC(Metric):
    """Computes TPR-FPR pairs for different thresholds. Works for both binary and
    multiclass problems. In the case of multiclass, the values will be calculated based
    on a one-vs-the-rest approach.

    Computation is performed in constant-memory by computing TPR and FPR for
    ``thresholds`` buckets/thresholds (evenly distributed between 0 and 1).

    Adapted from torchmetrics.BinnedPrecisionRecallCurve

    Forward accepts

    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass)
      tensor with probabilities, where C is the number of classes.

    - ``target`` (long tensor): ``(N, ...)`` or ``(N, C, ...)`` with integer labels

    Args:
        num_classes: integer with number of classes. For binary, set to 1.
        thresholds: list or tensor with specific thresholds or a number of bins from
            linear sampling.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``thresholds`` is not a ``int``, ``list`` or ``tensor``
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    TPs: torch.Tensor
    FPs: torch.Tensor
    TNs: torch.Tensor
    FNs: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        thresholds: Union[int, torch.Tensor, List[float], None] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        if isinstance(thresholds, int):
            self.num_thresholds = thresholds
            thresholds = torch.linspace(0, 1.0, thresholds)
            self.register_buffer("thresholds", thresholds)
        elif thresholds is not None:
            if not isinstance(thresholds, (list, torch.Tensor)):
                raise ValueError(
                    "Expected argument `thresholds` to either be an integer,"
                    " list of floats or a tensor"
                )
            thresholds = (
                torch.tensor(thresholds) if isinstance(thresholds, list) else thresholds
            )
            self.num_thresholds = thresholds.numel()
            self.register_buffer("thresholds", thresholds)

        for name in ("TPs", "FPs", "TNs", "FNs"):
            self.add_state(
                name=name,
                default=torch.zeros(
                    num_classes, self.num_thresholds, dtype=torch.float32
                ),
                dist_reduce_fx="sum",
            )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Args
            preds: (n_samples, n_classes) tensor
            target: (n_samples, n_classes) tensor
        """
        # binary case
        if len(preds.shape) == len(target.shape) == 1:
            preds = preds.reshape(-1, 1)
            target = target.reshape(-1, 1)

        if len(preds.shape) == len(target.shape) + 1:
            target = tmutils.to_onehot(target, num_classes=self.num_classes)

        target = target == 1
        # Iterate one threshold at a time to conserve memory
        for i in range(self.num_thresholds):
            predictions = preds >= self.thresholds[i]  # type: ignore
            self.TPs[:, i] += (target & predictions).sum(dim=0)
            self.FPs[:, i] += ((~target) & (predictions)).sum(dim=0)
            self.TNs[:, i] += ((~target) & (~predictions)).sum(dim=0)
            self.FNs[:, i] += ((target) & (~predictions)).sum(dim=0)

    def compute(
        self,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]],
    ]:
        """Returns float tensor of size n_classes."""
        TPR = (self.TPs + tmutils.METRIC_EPS) / (
            self.TPs + self.FNs + tmutils.METRIC_EPS
        )
        FPR = (self.FPs + tmutils.METRIC_EPS) / (
            self.FPs + self.TNs + tmutils.METRIC_EPS
        )

        # Need to guarantee that last TPR=1 and FPR=1, similar to precision_recall_curve
        t_ones = torch.ones(self.num_classes, 1, dtype=TPR.dtype, device=TPR.device)
        TPR = torch.cat([TPR, t_ones], dim=1)
        FPR = torch.cat([FPR, t_ones], dim=1)
        if self.num_classes == 1:
            return TPR[0, :], FPR[0, :], self.thresholds  # type: ignore
        return (
            list(TPR),
            list(FPR),
            [self.thresholds for _ in range(self.num_classes)],  # type: ignore
        )


class BinnedAUROC(BinnedROC):
    """Compute Area Under the Receiver Operating Characteristic Curve (`ROC AUC`_).
    Works for both binary, multilabel and multiclass problems. In the case of
    multiclass, the values will be calculated based on a one-vs-the-rest approach.

    Computation is performed in constant-memory by computing TPR and FPR for
    ``thresholds`` buckets/thresholds (evenly distributed between 0 and 1).

    Forward accepts

    - ``preds`` (float tensor): ``(N, ...)`` (binary) or ``(N, C, ...)`` (multiclass)
      tensor with probabilities, where C is the number of classes.

    - ``target`` (long tensor): ``(N, ...)`` or ``(N, C, ...)`` with integer labels

    Args:
        num_classes: integer with number of classes. For binary, set to 1.
        thresholds: list or tensor with specific thresholds or a number of bins from
            linear sampling.
        max_fpr:
            If not ``None``, calculates standardized partial AUC over the
            range ``[0, max_fpr]``. Should be a float between 0 and 1.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``thresholds`` is not a ``int``, ``list`` or ``tensor``
    """

    higher_is_better: Optional[bool] = True

    def __init__(
        self,
        num_classes: int,
        thresholds: Union[int, torch.Tensor, List[float], None] = None,
        max_fpr: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(num_classes, thresholds, **kwargs)
        self.max_fpr = max_fpr

        if self.max_fpr is not None:
            if not isinstance(max_fpr, float) or not 0 < max_fpr <= 1:
                raise ValueError(
                    f"`max_fpr` should be a float in range (0, 1], got: {max_fpr}"
                )

            if _TORCH_LOWER_1_6:
                raise RuntimeError(
                    "`max_fpr` argument requires `torch.bucketize` "
                    "which is not available below PyTorch version 1.6"
                )

    def compute(self) -> Union[List[torch.Tensor], torch.Tensor]:  # type: ignore
        tpr, fpr, _ = super().compute()
        if self.max_fpr is not None:
            tpr, fpr = (
                self._normalize_to_max_fpr(tpr, fpr)  # type: ignore
                if self.num_classes == 1
                else [self._normalize_to_max_fpr(t, f) for t, f in zip(tpr, fpr)]
            )

        auroc = (
            tm_compute_auc(fpr, tpr, reorder=True)  # type: ignore
            if self.num_classes == 1
            else [tm_compute_auc(f, t, reorder=True) for f, t in zip(fpr, tpr)]
        )

        if self.max_fpr is not None:
            auroc = (
                self._mcclish_correction(auroc)  # type: ignore
                if self.num_classes == 1
                else [self._mcclish_correction(a) for a in auroc]
            )

        return auroc

    def _normalize_to_max_fpr(
        self, tpr: torch.Tensor, fpr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_area = torch.tensor(self.max_fpr, device=fpr.device)

        # Add a single point at max_fpr and interpolate its tpr value
        stop = torch.bucketize(max_area, fpr, out_int32=True, right=True)
        weight = (max_area - fpr[stop - 1]) / (fpr[stop] - fpr[stop - 1])
        interp_tpr = torch.lerp(tpr[stop - 1], tpr[stop], weight)

        tpr = torch.cat([tpr[:stop], interp_tpr.view(1)])
        fpr = torch.cat([fpr[:stop], max_area.view(1)])
        return tpr, fpr

    def _mcclish_correction(self, auroc: torch.Tensor) -> torch.Tensor:
        """McClish correction: standardize result to be
        0.5 if non-discriminant and 1 if maximal
        """
        max_area = torch.tensor(self.max_fpr, device=auroc.device)
        min_area = 0.5 * max_area**2
        return 0.5 * (1 + (auroc - min_area) / (max_area - min_area))
