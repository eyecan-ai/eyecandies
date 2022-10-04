import typing as t

from pipelime.stages import SampleStage
from pydantic import Field

if t.TYPE_CHECKING:
    import torch
    from pipelime.sequences import Sample


class ComputeStatsStage(SampleStage, title="stats"):
    """Computes TP, FP, TN, FN statistics for each sample."""

    hm_thresholds: t.Optional[t.List[float]] = Field(
        ...,
        description=(
            "The thresholds to use to compute the heatmap statistics. Set to "
            "an empty list to use all the values or None to skip the computation."
        ),
    )
    score_thresholds: t.Optional[t.List[float]] = Field(
        ...,
        description=(
            "The thresholds to use to compute the anomaly score statistics. Set to "
            "an empty list to use all the values or None to skip the computation."
        ),
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

        if (
            self.hm_thresholds is not None
            and self.heatmap_key in x
            and self.mask_key in x
        ):
            hm: np.ndarray = x[self.heatmap_key]().squeeze()  # type: ignore
            mask = (x[self.mask_key]().squeeze() != 0).astype(np.int32)  # type: ignore

            # resize the mask to the heatmap size
            if mask.shape != hm.shape:
                mask = cv2.resize(mask, hm.shape[::-1], interpolation=cv2.INTER_NEAREST)

            # flatten, so that each pixel is a sample
            preds = torch.tensor(hm.reshape(-1))
            targets = torch.tensor(mask.reshape(-1))

            if len(self.hm_thresholds) > 0:
                x = self._update_binnedroc(
                    x, preds, targets, self.hm_thresholds, self.hm_stat_key_format
                )
            else:
                x = self._update_roc(x, preds, targets, self.hm_stat_key_format)

        if (
            self.score_thresholds is not None
            and self.anomaly_score_key in x
            and self.label_key in x
        ):
            preds = torch.tensor(x[self.anomaly_score_key]())
            targets = torch.tensor(x[self.label_key]().astype(np.int32))  # type: ignore

            if len(self.score_thresholds) > 0:
                x = self._update_binnedroc(
                    x, preds, targets, self.score_thresholds, self.score_stat_key_format
                )
            else:
                x = self._update_roc(x, preds, targets, self.score_stat_key_format)

        return x

    def _update_roc(
        self,
        x: "Sample",
        preds: "torch.Tensor",
        targets: "torch.Tensor",
        key_format: str,
    ) -> "Sample":
        from torchmetrics import ROC  # type: ignore
        from pipelime.items import NpyNumpyItem

        roc = ROC()
        roc.update(preds, targets)
        x = x.set_item(
            key_format.replace("*", "preds"),
            NpyNumpyItem([tn.numpy() for tn in roc.preds]),  # type: ignore
        )
        x = x.set_item(
            key_format.replace("*", "target"),
            NpyNumpyItem([tn.numpy() for tn in roc.target]),  # type: ignore
        )
        return x

    def _update_binnedroc(
        self,
        x: "Sample",
        preds: "torch.Tensor",
        targets: "torch.Tensor",
        thresholds: t.List[float],
        key_format: str,
    ) -> "Sample":
        from pipelime.items import NpyNumpyItem
        from eyecandies.modules.binned_roc import BinnedROC

        roc = BinnedROC(num_classes=1, thresholds=thresholds)
        roc.update(preds, targets)
        x = x.set_item(
            key_format.replace("*", "TPs"),
            NpyNumpyItem(roc.TPs.numpy()),
        )
        x = x.set_item(
            key_format.replace("*", "FPs"),
            NpyNumpyItem(roc.FPs.numpy()),
        )
        x = x.set_item(
            key_format.replace("*", "TNs"),
            NpyNumpyItem(roc.TNs.numpy()),
        )
        x = x.set_item(
            key_format.replace("*", "FNs"),
            NpyNumpyItem(roc.FNs.numpy()),
        )
        return x
