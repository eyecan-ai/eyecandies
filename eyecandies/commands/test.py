import typing as t
from pathlib import Path

from eyecandies.commands.utils import DataLoaderOptions, image_tensor_to_numpy
from pipelime.commands.interfaces import InputDatasetInterface, OutputDatasetInterface
from pipelime.piper import PipelimeCommand, PiperPortType
from pydantic import Field, PositiveInt


class TestCommand(PipelimeCommand, title="autoenc-test"):
    """Simple computation of predictions and metrics with a naive autoencoder."""

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
    predictions: t.Optional[OutputDatasetInterface] = OutputDatasetInterface.pyd_field(
        is_required=False,
        description="The output dataset with predictions and metrics.",
        piper_port=PiperPortType.OUTPUT,
    )
    final_metrics: t.Optional[t.Mapping[str, float]] = Field(
        None,
        description="Metrics computed on the test dataset.",
        exclude=True,
        repr=False,
        piper_port=PiperPortType.OUTPUT,
    )

    # PARAMETERS
    transforms: t.Optional[t.Mapping] = Field(
        None,
        description=(
            "Transformations to apply to input images (albumentation format)."
        ),
    )
    dataloader: DataLoaderOptions = Field(
        default_factory=DataLoaderOptions,  # type: ignore
        description="Torch data loader options.",
    )
    device: str = Field("cuda", description="The device to use for testing.")
    image_key: str = Field("image", description="The key of the image in the dataset.")
    image_size: PositiveInt = Field(
        256,
        description=(
            "Images will be rescaled to this size just before feeding the network."
        ),
    )
    image_channels: PositiveInt = Field(
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
        model.train(False)

        # create the metric
        auroc_mt = tm.AUROC()
        auroc_mt.to(device=device)

        # output dataset
        if self.predictions is not None:
            outpipe = self.predictions.as_pipe()
            if outpipe["to_underfolder"]["zfill"] is None:
                outpipe["to_underfolder"]["zfill"] = test_seq.best_zfill()
            out_stream = DataStream(output_pipe=outpipe)

        with torch.no_grad():
            for batch in self.track(data_loader, message="Test batches"):
                images = batch[self.image_key].to(device=device)
                labels = batch["label"].to(device=device)
                idxs = batch["~idx"]

                outputs = model(images)
                max_diffs = torch.abs(images - outputs).max(dim=1).values
                scores = torch.clamp(
                    torch.max(max_diffs.reshape(max_diffs.shape[0], -1), dim=1).values,
                    min=0.0,
                    max=1.0,
                )
                scores = scores.reshape(-1, 1)

                auroc_mt.update(scores, labels)

                if self.predictions is not None:
                    for img, index, hm, max_diff, pred in zip(
                        images, idxs, max_diffs, scores, outputs
                    ):
                        out_stream.set_output(  # type: ignore
                            idx=int(index.item()),
                            sample=Sample(
                                {
                                    "image": pli.PngImageItem(
                                        image_tensor_to_numpy(img)
                                    ),
                                    "output": pli.PngImageItem(
                                        image_tensor_to_numpy(pred)
                                    ),
                                    "heatmap": pli.NpyNumpyItem(
                                        image_tensor_to_numpy(hm, False)
                                    ),
                                    "heatmap_img": pli.PngImageItem(
                                        image_tensor_to_numpy(hm)
                                    ),
                                    "score": pli.TxtNumpyItem(
                                        float(max_diff.cpu().item())
                                    ),
                                }
                            ),
                        )

        auroc = float(auroc_mt.compute().cpu().numpy())
        self.final_metrics = {"AUROC": auroc}

        if self.predictions is not None:
            out_stream.set_output(  # type: ignore
                idx=0, sample=Sample({"auroc": pli.TxtNumpyItem(auroc, shared=True)})
            )
