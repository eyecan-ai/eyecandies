import typing as t

from pipelime.stages import SampleStage
from pydantic import Field

if t.TYPE_CHECKING:
    from pipelime.sequences import Sample


class ComputeAnomalyScoreStage(SampleStage, title="ascore"):
    """Computes the anomaly score from the heatmap."""

    heatmap_key: str = Field("heatmap", description="The key of the heatmap.")
    anomaly_score_key: str = Field(
        "anomaly_score", description="The key of the anomaly score."
    )

    def __call__(self, x: "Sample") -> "Sample":
        from pipelime.items import TxtNumpyItem

        if self.anomaly_score_key not in x and self.heatmap_key in x:
            heatmap = x[self.heatmap_key]()
            anomaly_score = heatmap.max()  # type: ignore
            x = x.set_item(self.anomaly_score_key, TxtNumpyItem(anomaly_score))
        return x


class AnomalyLabelFromMask(SampleStage, title="gt-label"):
    """Adds a groundtruth anomaly label of 0 if the mask is all 0, 1 otherwise."""

    mask_key: str = Field("mask", description="The key of the mask.")
    label_key: str = Field("target", description="The key of the label.")

    def __call__(self, x: "Sample") -> "Sample":
        from pipelime.items import TxtNumpyItem

        if self.label_key not in x and self.mask_key in x:
            mask = x[self.mask_key]()
            label = 1 if mask.any() else 0  # type: ignore
            x = x.set_item(self.label_key, TxtNumpyItem(label))
        return x


class ComputeMinMax(SampleStage, title="minmax"):
    """Compute min-max value of the heatmaps."""

    heatmap_key: str = Field("heatmap", description="The key of the heatmap.")
    minmax_key: str = Field("minmax", description="The key of the min-max values.")

    def __call__(self, x: "Sample") -> "Sample":
        from pipelime.items import TxtNumpyItem

        if self.minmax_key not in x and self.heatmap_key in x:
            heatmap = x[self.heatmap_key]()
            x = x.set_item(
                self.minmax_key,
                TxtNumpyItem(
                    [float(heatmap.min()), float(heatmap.max())]  # type: ignore
                ),
            )
        return x
