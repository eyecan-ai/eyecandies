from eyecandies.stages.add_label import AddLabelStage, BadLabelStage, GoodLabelStage
from eyecandies.stages.anomaly_score import (
    ComputeAnomalyScoreStage,
    AnomalyLabelFromMask,
    ComputeMinMax,
)
from eyecandies.stages.depth import DepthToMetersStage, DepthToPCStage
from eyecandies.stages.stats import ComputeStatsStage
