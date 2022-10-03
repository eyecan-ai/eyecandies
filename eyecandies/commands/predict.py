import typing as t
from pathlib import Path

from eyecandies.commands.utils import DataLoaderOptions, image_tensor_to_numpy
from pipelime.commands.interfaces import InputDatasetInterface, OutputDatasetInterface
from pipelime.piper import PipelimeCommand, PiperPortType
from pydantic import Field


class PredictCommand(PipelimeCommand, title="autoenc-predict"):
    """Compute predictions on a test dataset with a naive autoencoder."""

    # INPUT
    test_dataset: InputDatasetInterface = InputDatasetInterface.pyd_field(
        description="Test dataset.", piper_port=PiperPortType.INPUT
    )
    ckpt: Path = Field(
        "last.ckpt",
        description="The checkpoint to load.",
        piper_port=PiperPortType.INPUT,
    )

    # OUTPUT
    predictions: OutputDatasetInterface = OutputDatasetInterface.pyd_field(
        description="The output dataset with predictions.",
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
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        import pipelime.stages as plst
        import pipelime.items as pli
        from pipelime.sequences import Sample
        from pipelime.sequences.utils import DataStream
        from eyecandies.modules.autoencoder import AutoEncoder

        test_seq = (
            self.test_dataset.create_reader()
            .map(plst.StageKeysFilter(key_list=[self.image_key], negate=False))
            .enumerate()
        )

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

        # output dataset
        outpipe = self.predictions.as_pipe()
        if outpipe["to_underfolder"]["zfill"] is None:
            outpipe["to_underfolder"]["zfill"] = test_seq.best_zfill()
        out_stream = DataStream(output_pipe=outpipe)

        with torch.no_grad():
            for batch in self.track(data_loader, message="Test batches"):
                images = batch[self.image_key].to(device=device)
                idxs = batch["~idx"]

                outputs = model(images)
                max_diffs = torch.abs(images - outputs).max(dim=1).values
                scores = torch.clamp(
                    torch.max(max_diffs.reshape(max_diffs.shape[0], -1), dim=1).values,
                    min=0.0,
                    max=1.0,
                )

                for img, index, hm, max_diff, pred in zip(
                    images, idxs, max_diffs, scores, outputs
                ):
                    out_stream.set_output(  # type: ignore
                        idx=int(index.item()),
                        sample=Sample(
                            {
                                "image": pli.PngImageItem(image_tensor_to_numpy(img)),
                                "output": pli.PngImageItem(image_tensor_to_numpy(pred)),
                                "heatmap": pli.NpyNumpyItem(
                                    image_tensor_to_numpy(hm, False)
                                ),
                                "heatmap_img": pli.PngImageItem(
                                    image_tensor_to_numpy(hm)
                                ),
                                "score": pli.TxtNumpyItem(float(max_diff.cpu().item())),
                            }
                        ),
                    )
