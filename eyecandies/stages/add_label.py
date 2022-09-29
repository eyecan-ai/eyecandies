import typing as t

from pipelime.stages import SampleStage
from pydantic import Field

if t.TYPE_CHECKING:
    from pipelime.sequences import Sample


class AddLabelStage(SampleStage, title="add-label"):
    """Adds an integer label to the sample."""

    label: int = Field(..., description="The label to add.")
    label_key: str = Field("label", description="The key of the label in the sample.")

    def __call__(self, x: "Sample") -> "Sample":
        from pipelime.items import TxtNumpyItem

        return x.set_item(self.label_key, TxtNumpyItem([self.label]))


class GoodLabelStage(AddLabelStage, title="good-label"):
    """Adds label 0 to the samples."""

    def __init__(self, **data):
        super().__init__(label=0, **data)


class BadLabelStage(AddLabelStage, title="bad-label"):
    """Adds label 1 to the samples."""

    def __init__(self, **data):
        super().__init__(label=1, **data)
