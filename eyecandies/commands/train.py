import typing as t
from pathlib import Path

from pipelime.piper import PipelimeCommand, PiperPortType
from pipelime.commands.interfaces import InputDatasetInterface
from pydantic import Field, PositiveInt, PositiveFloat

from eyecandies.commands.utils import DataLoaderOptions


class TrainCommand(PipelimeCommand, title="ec-train"):
    """Train a simple autoencoder."""

    # INPUT
    train_dataset: InputDatasetInterface = InputDatasetInterface.pyd_field(
        description="Train dataset.", piper_port=PiperPortType.INPUT
    )

    # OUTPUT
    last_ckpt: t.Optional[Path] = Field(
        "last.ckpt",
        description="Where to save the last checkpoint of the training.",
        piper_port=PiperPortType.OUTPUT,
    )
    training_stats: t.Optional[t.Mapping[str, float]] = Field(
        None,
        description="Final training statistics.",
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
    repeat: PositiveInt = Field(
        1,
        description=(
            "Number of times to repeat the dataset before applying the transformations."
        ),
    )
    dataloader: DataLoaderOptions = Field(
        default_factory=(lambda: DataLoaderOptions(shuffle=True)),  # type: ignore
        description="Torch data loader options.",
    )
    device: str = Field("cuda", description="The device to use for training.")
    n_epochs: PositiveInt = Field(10, description="Number of epochs to train for.")
    learning_rate: PositiveFloat = Field(
        1e-3, description="Learning rate of the optimizer."
    )
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
        import pipelime.stages as plst
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import torch
        from torch import nn, optim

        from eyecandies.modules.autoencoder import AutoEncoder

        # create a reader
        train_seq = self.train_dataset.create_reader().repeat(self.repeat)

        # keep only the image item
        train_seq = train_seq.map(
            plst.StageKeysFilter(key_list=[self.image_key], negate=False)
        )

        # apply the user transformations, if any
        if self.transforms is not None:
            train_seq = train_seq.map(
                plst.StageAlbumentations(
                    transform=self.transforms, keys_to_targets={self.image_key: "image"}
                )
            )

        # resize the images, normalize and convert them to tensors
        train_seq = train_seq.map(
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
        data_loader = self.dataloader.make_dataloader(train_seq)

        # create the model
        device = torch.device(self.device)
        model = AutoEncoder(
            image_size=self.image_size, image_channels=self.image_channels
        )
        model.train(True)
        model.to(device=device)

        # optimizer and loss function
        loss_fn = nn.L1Loss(reduction="mean")
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        current_loss = 0
        for epoch in self.track(range(self.n_epochs), message="Training epochs"):
            for batch in data_loader:
                images = batch[self.image_key].to(device=device)

                optimizer.zero_grad()

                output = model(images)
                loss = loss_fn(output, images)
                loss.backward()
                optimizer.step()

                current_loss = float(loss.detach().cpu().item())

        self.training_stats = {"loss": current_loss}
        if self.last_ckpt is not None:
            self.last_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(self.last_ckpt))
