import typing as t
from pathlib import Path

from eyecandies.commands.utils import DataLoaderOptions
from pipelime.commands.interfaces import InputDatasetInterface
from pipelime.piper import PipelimeCommand, PiperPortType
from pydantic import Field


class TestCommand(PipelimeCommand, title="ec-test"):
    """Compute predictions and metrics."""

    # INPUT
    good_dataset: InputDatasetInterface = InputDatasetInterface.pyd_field(
        description="Test dataset of `good` samples.", piper_port=PiperPortType.INPUT
    )
    bad_dataset: InputDatasetInterface = InputDatasetInterface.pyd_field(
        description="Test dataset of `bad` samples.", piper_port=PiperPortType.INPUT
    )
    ckpt: Path = Field(
        "last.ckpt",
        description="The checkpoint to load.",
        piper_port=PiperPortType.INPUT,
    )

    # OUTPUT
    predictions: t.Optional[Path] = Field(
        None,
        description="The output dataset with predictions and metrics.",
        piper_port=PiperPortType.OUTPUT,
    )
    final_metrics: t.Optional[t.Mapping[str, float]] = Field(
        None,
        exclude=True,
        repr=False,
        piper_port=PiperPortType.OUTPUT,
    )

    # PARAMETERS
    transforms: t.Optional[t.Mapping] = Field(
        None, description="Transformations to apply to input images."
    )
    dataloader: DataLoaderOptions = Field(
        default_factory=DataLoaderOptions,  # type: ignore
        description="Torch data loader options.",
    )
    device: str = Field("cuda", description="The device to use for testing.")
    image_key: str = Field("image", description="The key of the image in the dataset.")
    image_size: int = Field(
        256,
        description=(
            "Images will be rescaled to this size just before feeding the network."
        ),
    )
    image_channels: int = Field(
        3, description="Number of channels of the input images."
    )

    def run(self):
        import torch
        import torchmetrics as tm
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import pipelime.stages as plst
        import pipelime.items as pli
        from pipelime.sequences import Sample
        from pipelime.sequences.utils import DataStream
        from eyecandies.modules.autoencoder import AutoEncoder
        from eyecandies.stages import BadLabelStage, GoodLabelStage

        # concatenate good and bad samples
        test_seq = (
            self.good_dataset.create_reader()
            .map(plst.StageKeysFilter(key_list=[self.image_key], negate=False))
            .map(GoodLabelStage())
        )
        test_seq += (
            self.bad_dataset.create_reader()
            .map(plst.StageKeysFilter(key_list=[self.image_key], negate=False))
            .map(BadLabelStage())
        )
        test_seq = test_seq.enumerate()

        # apply the user transformations, if any
        if self.transforms is not None:
            test_seq = test_seq.map(
                plst.StageAlbumentations(
                    transform=self.transforms, keys_to_targets={self.image_key: "image"}
                )
            )

        # resize the images and convert them to tensors
        test_seq = test_seq.map(
            plst.StageAlbumentations(
                transform=A.Compose(
                    [
                        A.Resize(self.image_size, self.image_size),
                        A.Normalize(mean=0.0, std=1.0, always_apply=True),
                        ToTensorV2(),
                    ]
                ),
                keys_to_targets={self.image_key: "image"},
            )
        )

        # create the data loader
        data_loader = self.dataloader.make_dataloader(test_seq)

        # create the model
        device = torch.device(self.device)
        model = AutoEncoder(
            image_size=self.image_size, image_channels=self.image_channels
        )
        model.to(device=device)
        model.load_state_dict(torch.load(self.ckpt))

        # create the metric
        auroc_mt = tm.AUROC()
        auroc_mt.to(device=device)

        # output dataset
        if self.predictions is not None:
            out_stream = DataStream.create_new_underfolder(
                str(self.predictions), zfill=test_seq.best_zfill()
            )

        with torch.no_grad():
            for batch in self.track(data_loader, message="Test batches"):
                images = batch[self.image_key].to(device=device)
                labels = batch["label"].to(device=device)
                idxs = batch["~idx"]

                output = model(images)
                diff = torch.abs(images - output)
                scores = torch.clamp(
                    torch.max(diff.reshape(diff.shape[0], -1), dim=1).values,
                    min=0.0,
                    max=1.0,
                )

                auroc_mt(labels, scores)

                if self.predictions is not None:
                    for img, i, d, s, pred in zip(images, idxs, diff, scores, output):
                        out_stream.set_output(  # type: ignore
                            idx=i,
                            sample=Sample(
                                {
                                    "image": pli.PngImageItem(img),
                                    "output": pli.PngImageItem(pred),
                                    "diff": pli.PngImageItem(d),
                                    "score": pli.TxtNumpyItem(s),
                                }
                            ),
                        )

        auroc = float(auroc_mt.compute().cpu().numpy())
        self.final_metrics = {"AUROC": auroc}

        if self.predictions is not None:
            out_stream.set_output(  # type: ignore
                idx=0, sample=Sample({"auroc": pli.TxtNumpyItem(auroc, shared=True)})
            )
