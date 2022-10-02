import typing as t

from pipelime.stages import SampleStage
from pydantic import Field

if t.TYPE_CHECKING:
    from pipelime.sequences import Sample


class ComputeStatsStage(SampleStage, title="stats"):
    """Computes TP, FP, TN, FN statistics for each sample."""

    hm_thresholds: t.Optional[t.List[float]] = Field(
        ..., description="The thresholds to use to compute the heatmap statistics."
    )
    score_thresholds: t.Optional[t.List[float]] = Field(
        ...,
        description="The thresholds to use to compute the anomaly score statistics.",
    )

    heatmap_key: str = Field("heatmap", description="The key of the heatmap.")
    anomaly_score_key: str = Field(
        "anomaly_score", description="The key of the anomaly score."
    )
    mask_key: str = Field("mask", description="The key of the groundtruth mask.")
    label_key: str = Field("label", description="The key of the groundtruth label.")

    hm_stat_key_format: str = Field(
        "hm_*",
        description=(
            "The key format of the heatmap statistics. "
            "Any `*` will be replaced by `TP`, `FP`..."
        ),
    )
    score_stat_key_format: str = Field(
        "score_*",
        description=(
            "The key format of the heatmap statistics. "
            "Any `*` will be replaced by `TP`, `FP`..."
        ),
    )

    def __call__(self, x: "Sample") -> "Sample":
        import torch
        import numpy as np
        import cv2
        from pipelime.items import TxtNumpyItem
        from eyecandies.modules.binned_roc import BinnedROC

        if (
            self.hm_thresholds is not None
            and self.heatmap_key in x
            and self.mask_key in x
        ):
            hm: np.ndarray = x[self.heatmap_key]()  # type: ignore
            mask = (x[self.mask_key]() != 0).astype(np.int32)

            if mask.shape != hm.shape:
                mask = cv2.resize(mask, hm.shape[::-1], interpolation=cv2.INTER_NEAREST)

            hm_roc = BinnedROC(num_classes=1, thresholds=self.hm_thresholds)
            preds = torch.tensor(hm.reshape(-1))
            targets = torch.tensor(mask.reshape(-1))
            hm_roc.update(preds, targets)
            x = x.set_item(
                self.hm_stat_key_format.replace("*", "TP"),
                TxtNumpyItem(hm_roc.TPs.numpy()),
            )
            x = x.set_item(
                self.hm_stat_key_format.replace("*", "FP"),
                TxtNumpyItem(hm_roc.FPs.numpy()),
            )
            x = x.set_item(
                self.hm_stat_key_format.replace("*", "TN"),
                TxtNumpyItem(hm_roc.TNs.numpy()),
            )
            x = x.set_item(
                self.hm_stat_key_format.replace("*", "FN"),
                TxtNumpyItem(hm_roc.FNs.numpy()),
            )

        if self.score_thresholds is not None and self.anomaly_score_key in x:
            score_roc = BinnedROC(num_classes=1, thresholds=self.score_thresholds)
            preds = torch.tensor(x[self.anomaly_score_key]())
            targets = torch.tensor(x[self.label_key]().astype(np.int32))  # type: ignore
            score_roc.update(preds, targets)
            x = x.set_item(
                self.score_stat_key_format.replace("*", "TP"),
                TxtNumpyItem(score_roc.TPs.numpy()),
            )
            x = x.set_item(
                self.score_stat_key_format.replace("*", "FP"),
                TxtNumpyItem(score_roc.FPs.numpy()),
            )
            x = x.set_item(
                self.score_stat_key_format.replace("*", "TN"),
                TxtNumpyItem(score_roc.TNs.numpy()),
            )
            x = x.set_item(
                self.score_stat_key_format.replace("*", "FN"),
                TxtNumpyItem(score_roc.FNs.numpy()),
            )

        return x
