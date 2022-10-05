import typing as t
from pathlib import Path
from abc import ABC, abstractmethod

from pipelime.commands.interfaces import InputDatasetInterface, GrabberInterface
from pipelime.piper import PipelimeCommand, PiperPortType
from pydantic import BaseModel, Field, PositiveInt


class _StatAggregator(ABC):
    def __init__(self, format_key: str):
        self.format_key = format_key
        self.buffers = {}

    def is_valid(self):
        return all(v is not None for v in self.buffers.values())

    def update(self, x):
        for name, current in self.buffers.items():
            curr_key = self.format_key.replace("*", name)
            if curr_key in x:
                value = x[curr_key]()
                self.buffers[name] = self._internal_update(current, value)

    @abstractmethod
    def compute_roc(self):
        pass

    @abstractmethod
    def compute_auroc(self, max_fpr):
        pass

    def _add_buffers(self, names: t.Sequence[str]):
        for n in names:
            self.buffers[n] = None

    @abstractmethod
    def _internal_update(self, current, value):
        pass

    def _internal_compute(self, metric):
        import warnings

        for name, value in self.buffers.items():
            setattr(metric, name, value)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message=".*was called before the.*",
                category=UserWarning,
            )
            return metric.compute()


class _PlainStatAggregator(_StatAggregator):
    def __init__(self, format_key: str):
        super().__init__(format_key)
        self._add_buffers(["preds", "target"])

    def _internal_update(self, current, value):
        import torch

        value = [torch.tensor(v) for v in value]
        return value if current is None else current + value

    def compute_roc(self):
        import warnings
        from torchmetrics import ROC  # type: ignore

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message=".*will save all targets and predictions in buffer.*",
                category=UserWarning,
            )
            roc = ROC(num_classes=1, pos_label=1)
        return self._internal_compute(roc)

    def compute_auroc(self, max_fpr):
        import warnings
        from torchmetrics import AUROC  # type: ignore
        from torchmetrics.utilities.enums import DataType

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message=".*will save all targets and predictions in buffer.*",
                category=UserWarning,
            )
            auroc = AUROC(num_classes=1, pos_label=1, max_fpr=max_fpr)
        auroc.mode = DataType.BINARY
        return self._internal_compute(auroc)


class _BinnedStatAggregator(_StatAggregator):
    def __init__(self, format_key: str, thresholds: t.Optional[t.List[float]]):
        super().__init__(format_key)
        self._thresholds = thresholds
        self._add_buffers(["TPs", "FPs", "TNs", "FNs"])

    def _internal_update(self, current, value):
        import torch

        value = torch.tensor(value)
        return value if current is None else current + value

    def compute_roc(self):
        from eyecandies.modules.binned_roc import BinnedROC

        return self._internal_compute(
            BinnedROC(num_classes=1, thresholds=self._thresholds)
        )

    def compute_auroc(self, max_fpr):
        from eyecandies.modules.binned_roc import BinnedAUROC

        return self._internal_compute(
            BinnedAUROC(num_classes=1, thresholds=self._thresholds, max_fpr=max_fpr)
        )


class MappedOutput(BaseModel):
    data: t.Mapping[str, t.Any]

    def __repr__(self) -> str:
        return self.__piper_repr__()

    def __piper_repr__(self) -> str:
        from io import StringIO
        from rich.console import Console
        from rich.table import Table

        table = Table(show_header=False)
        for k, v in self.data.items():
            table.add_row(k, str(v))

        strout = StringIO()
        console = Console(file=strout)
        console.print(table)
        return strout.getvalue()


class ComputeMetricsCommand(PipelimeCommand, title="ec-metrics"):
    """Compute metrics on given predictions and groundtruth.
    The predictions should include a heatmap and/or a classification score.
    """

    # INPUT
    predictions: InputDatasetInterface = InputDatasetInterface.pyd_field(
        alias="p",
        description="The dataset with predictions.",
        piper_port=PiperPortType.INPUT,
    )
    targets: InputDatasetInterface = InputDatasetInterface.pyd_field(
        alias="t",
        description="The dataset with groundtruth labels and masks.",
        piper_port=PiperPortType.INPUT,
    )

    # OUTPUT
    output_folder: Path = Field(
        ...,
        alias="o",
        description=(
            "The folder where the output CSV file will be written. "
            "It will be created if it does not exist."
        ),
        piper_port=PiperPortType.OUTPUT,
    )
    output_global_stats: t.Optional[MappedOutput] = Field(
        None,
        description="Final global statistics.",
        exclude=True,
        repr=False,
        piper_port=PiperPortType.OUTPUT,
    )

    # PARAMETERS
    output_prefix: str = Field("", description="The prefix of the output CSV file.")
    heatmap_key: t.Optional[str] = Field(
        "heatmap",
        description=(
            "The key of the heatmap in the dataset. The pixel metrics will be computed "
            "by thresholding its values, while the maximum will be used to compute "
            "the image metrics if the `anomaly_score_key` is not given or not found."
        ),
    )
    anomaly_score_key: t.Optional[str] = Field(
        None,
        description=(
            "The key of the anomaly score in the dataset. If valid, it will be used "
            "to compute the image metrics, instead of the maximum value of the heatmap."
        ),
    )

    mask_key: str = Field(
        "mask", description="The key of the mask in the target dataset."
    )

    nbins: t.Optional[PositiveInt] = Field(
        None,
        alias="b",
        description=(
            "Compute the ROC curve at `nbins` thresholds linearly sampled "
            "between the minimum and maximum values. "
            "Leave empty to use all the values "
            "(WARNING: this may lead to large memory footprint)."
        ),
    )

    pixel_auroc_max_fpr: t.Optional[float] = Field(
        None,
        description=(
            "Compute the pixel AUROC on a reduced range. "
            "Leave empty to use the whole curve."
        ),
    )
    image_auroc_max_fpr: t.Optional[float] = Field(
        None,
        description=(
            "Compute the image AUROC on a reduced range. "
            "Leave empty to use the whole curve."
        ),
    )

    grabber: GrabberInterface = GrabberInterface.pyd_field(alias="g")

    def run(self):  # noqa: C901
        import yaml
        import numpy as np
        import pipelime.stages as plst
        import eyecandies.stages as ecst

        pred_seq = self.predictions.create_reader()
        gt_seq = self.targets.create_reader()

        if len(pred_seq) != len(gt_seq):
            raise ValueError("Predictions and targets have different lengths.")

        # extract the relevant items from the predictions
        effective_keys = []
        if self.heatmap_key:
            effective_keys.append(self.heatmap_key)
        if self.anomaly_score_key:
            effective_keys.append(self.anomaly_score_key)
        if not effective_keys:
            raise ValueError(
                "You should provide the heatmap key and/or the anomaly score key."
            )
        pred_seq = pred_seq.map(
            plst.StageKeysFilter(key_list=effective_keys, negate=False)
        )

        anomaly_score_key = self.anomaly_score_key
        if self.anomaly_score_key not in effective_keys:
            anomaly_score_key = self._make_unique("anomaly_score", effective_keys)
            effective_keys.append(anomaly_score_key)
            pred_seq = pred_seq.map(
                ecst.ComputeAnomalyScoreStage(
                    heatmap_key=self.heatmap_key,  # type: ignore
                    anomaly_score_key=anomaly_score_key,
                )
            )

        # extract the relevant items from the targets
        gt_seq = gt_seq.map(
            plst.StageKeysFilter(key_list=[self.mask_key], negate=False)
        )

        gt_label_key = self._make_unique("label", effective_keys)
        effective_keys.append(gt_label_key)
        gt_seq = gt_seq.map(
            ecst.AnomalyLabelFromMask(mask_key=self.mask_key, label_key=gt_label_key)
        )

        mask_key = self._make_unique(self.mask_key, effective_keys)
        effective_keys.append(mask_key)
        if mask_key != self.mask_key:
            gt_seq = gt_seq.map(
                plst.StageRemap(remap={self.mask_key: mask_key}, remove_missing=False)
            )

        # zip the samples toghether and cache
        test_seq = pred_seq.zip(gt_seq).cache()

        hm_range, score_range = (np.inf, -np.inf), (np.inf, -np.inf)
        if self.nbins:
            # compute the ranges of the heatmaps and the anomaly scores
            hm_range_key = self._make_unique("hm_range", effective_keys)
            effective_keys.append(hm_range_key)

            def _update_ranges(x):
                nonlocal hm_range, score_range
                if self.heatmap_key:
                    hmr = x[hm_range_key]()
                    hm_range = (
                        min(hm_range[0], float(hmr[0])),
                        max(hm_range[1], float(hmr[1])),
                    )
                score_range = (
                    min(score_range[0], float(x[anomaly_score_key]())),
                    max(score_range[1], float(x[anomaly_score_key]())),
                )

            self.grabber.grab_all(
                test_seq.map(
                    ecst.ComputeMinMax(
                        heatmap_key=self.heatmap_key, minmax_key=hm_range_key
                    )
                )
                if self.heatmap_key
                else test_seq,
                keep_order=False,
                parent_cmd=self,
                track_message=f"Computing ranges ({len(test_seq)} samples)",
                sample_fn=_update_ranges,
            )

        if not self.heatmap_key:
            hm_thresholds = None
        elif self.nbins:
            hm_thresholds = np.linspace(hm_range[0], hm_range[1], self.nbins).tolist()
        else:
            hm_thresholds = []

        score_thresholds = (
            np.linspace(score_range[0], score_range[1], self.nbins).tolist()
            if self.nbins
            else []
        )

        # now compute TP, FP, TN, FN for each threshold and each sample
        # in each subprocess, then aggregate the results in the main process
        hm_stats = "__hm_stats__*"
        score_stats = "__score_stats__*"

        hm_stats_agg = (
            _BinnedStatAggregator(hm_stats, hm_thresholds)
            if self.nbins
            else _PlainStatAggregator(hm_stats)
        )
        score_stats_agg = (
            _BinnedStatAggregator(score_stats, score_thresholds)
            if self.nbins
            else _PlainStatAggregator(score_stats)
        )

        good_count, bad_count = 0, 0

        def _update_stats(x):
            nonlocal good_count, bad_count
            if x[gt_label_key]() == 0:
                bad_count += 1
            else:
                good_count += 1

            nonlocal hm_stats_agg, score_stats_agg
            hm_stats_agg.update(x)
            score_stats_agg.update(x)

        self.grabber.grab_all(
            test_seq.map(
                ecst.ComputeStatsStage(
                    hm_thresholds=hm_thresholds,
                    score_thresholds=score_thresholds,
                    heatmap_key=self.heatmap_key if self.heatmap_key else "",
                    anomaly_score_key=anomaly_score_key,  # type: ignore
                    mask_key=mask_key,
                    label_key=gt_label_key,
                    hm_stat_key_format=hm_stats,
                    score_stat_key_format=score_stats,
                )
            ),
            keep_order=False,
            parent_cmd=self,
            track_message=f"Computing statistics ({len(test_seq)} samples)",
            sample_fn=_update_stats,
        )

        # Write out the results
        global_meta = {
            "good_samples": good_count,
            "bad_samples": bad_count,
        }

        if hm_stats_agg.is_valid():
            global_meta = {
                **global_meta,
                "pixel_auroc": float(
                    hm_stats_agg.compute_auroc(
                        self.pixel_auroc_max_fpr
                    ).numpy()  # type: ignore
                ),
                "pixel_auroc_max_fpr": self.pixel_auroc_max_fpr,
            }

            tpr, fpr, thr = hm_stats_agg.compute_roc()
            with self._make_output("pixel_roc.csv").open("w") as f:
                f.write("TPR,FPR,Threshold\n")
                for trp_v, fpr_v, thr_v in zip(tpr, fpr, thr):
                    f.write(f"{float(trp_v)},{float(fpr_v)},{float(thr_v)}\n")

        if score_stats_agg.is_valid():
            global_meta = {
                **global_meta,
                "image_auroc": float(
                    score_stats_agg.compute_auroc(
                        self.image_auroc_max_fpr
                    ).numpy()  # type: ignore
                ),
                "image_auroc_max_fpr": self.image_auroc_max_fpr,
            }

            tpr, fpr, thr = score_stats_agg.compute_roc()
            with self._make_output("image_roc.csv").open("w") as f:
                f.write("TPR,FPR,Threshold\n")
                for trp_v, fpr_v, thr_v in zip(tpr, fpr, thr):
                    f.write(f"{float(trp_v)},{float(fpr_v)},{float(thr_v)}\n")

        self.output_global_stats = MappedOutput(data=global_meta)

    def _make_unique(self, key: str, key_list: t.List[str]) -> str:
        while key in key_list:
            key += "-"
        return key

    def _make_output(self, name: str):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        return self.output_folder / (self.output_prefix + name)
