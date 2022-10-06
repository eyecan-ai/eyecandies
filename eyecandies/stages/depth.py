import typing as t

from pipelime.stages import SampleStage
from pydantic import Field

if t.TYPE_CHECKING:
    from pipelime.sequences import Sample


class DepthToMetersStage(SampleStage, title="depth2mt"):
    """Converts 16bit depth images to meters as floating point numpy array."""

    depth_key: str = Field(
        "depth", description="The key of the input depth image item."
    )
    mind_key_address: str = Field(
        "info_depth.normalization.min",
        description="The pydash address of the minimum depth value in the sample's item.",
    )
    maxd_key_address: str = Field(
        "info_depth.normalization.max",
        description="The pydash address of the maximum depth value in the sample's item.",
    )

    use_float64: bool = Field(
        False, description="Whether to use 64-bit floats for the output metric depth."
    )

    out_depth_key_format: str = Field(
        "*",
        description="The name of the item containing the output depth image. Any `*` will be replaced with the input item key.",
    )

    def __call__(self, x: "Sample") -> "Sample":
        import numpy as np
        from pipelime.items import NpyNumpyItem

        mind = x.deep_get(self.mind_key_address)
        maxd = x.deep_get(self.maxd_key_address)
        if (
            self.depth_key not in x
            or not isinstance(mind, float)
            or not isinstance(maxd, float)
        ):
            return x

        dimg: np.ndarray = x[self.depth_key]()  # type: ignore
        dimg = dimg.astype(np.float64 if self.use_float64 else np.float32)
        dimg = dimg / 65535.0 * (maxd - mind) + mind

        x = x.set_item(
            self.out_depth_key_format.replace("*", self.depth_key), NpyNumpyItem(dimg)
        )
        return x
