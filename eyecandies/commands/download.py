import typing as t
from pathlib import Path

from pipelime.piper import PipelimeCommand, PiperPortType
from pydantic import Field, validator


class FileChunkIterator:
    def __init__(self, blocksize: int, response):
        self.blocksize = blocksize
        self.response = response

    def __iter__(self):
        return self

    def __next__(self):
        data = self.response.read(self.blocksize)
        if not data:
            raise StopIteration
        return data


class TarballIterator:
    def __init__(self, tarball):
        self.members = tarball.getmembers()

    def __iter__(self):
        for member in self.members:
            yield member


class GetEyecandiesCommand(PipelimeCommand, title="ec-get"):
    """Download the eyecandies dataset."""

    category: t.Union[str, t.Sequence[str], None] = Field(
        None,
        alias="c",
        description=(
            "One or more object categories to download "
            "(the name is case- and space-insensitive). "
            "Defaults to the whole dataset."
        ),
    )
    output: Path = Field(
        ...,
        alias="o",
        description="The output folder. It will be created if it does not exist.",
        piper_port=PiperPortType.OUTPUT,
    )

    @validator("output")
    def _validate_output(cls, v: Path):
        if v.exists() and not v.is_dir():
            raise ValueError(f"Output folder `{v}` is not a directory.")
        return v

    def run(self):
        from io import BytesIO
        import tarfile
        from urllib.request import urlopen

        for url, name in self._url_iterator():
            target = self.output / f"{''.join(name.split())}"
            if target.exists() and target.is_dir():
                raise ValueError(f"Output folder `{target}` already exists.")

        for url, name in self._url_iterator():
            resp = urlopen(url)
            length = resp.getheader("content-length")
            if length:
                length = int(length)
                blocksize = max(4096, length // 100)
            else:
                length = 2**32
                blocksize = 1000000

            buffer = BytesIO()
            for data in self.track(
                FileChunkIterator(blocksize, resp),
                size=(length + blocksize - 1) // blocksize,
                message=f"Downloading {name}",
            ):
                buffer.write(data)
            buffer.seek(0)

            target = self.output / f"{''.join(name.split())}"
            target.mkdir(parents=True, exist_ok=False)
            target = str(target)
            with tarfile.open(fileobj=buffer, mode="r") as tarball:
                for member in self.track(
                    TarballIterator(tarball),
                    size=len(tarball.getmembers()),
                    message=f"Extracting {name}",
                ):
                    tarball.extractall(path=target, members=[member])

    def _url_iterator(self):
        std_names = {
            "candycane": "Candy Cane",
            "chocolatecookie": "Chocolate Cookie",
            "chocolatepraline": "Chocolate Praline",
            "confetto": "Confetto",
            "gummybear": "Gummy Bear",
            "hazelnuttruffle": "Hazelnut Truffle",
            "licoricesandwich": "Licorice Sandwich",
            "lollipop": "Lollipop",
            "marshmallow": "Marshmallow",
            "peppermintcandy": "Peppermint Candy",
        }
        std_url = {
            "candycane": r"https://drive.google.com/uc?export=download&id=1ZEWNNoDp9NO_p_uqHwCaMH6L32BDkPm6",  # noqa: E501
            "chocolatecookie": r"https://drive.google.com/uc?export=download&id=1ZEWNNoDp9NO_p_uqHwCaMH6L32BDkPm6",  # noqa: E501
            "chocolatepraline": r"https://drive.google.com/uc?export=download&id=1ZEWNNoDp9NO_p_uqHwCaMH6L32BDkPm6",  # noqa: E501
            "confetto": r"https://drive.google.com/uc?export=download&id=1ZEWNNoDp9NO_p_uqHwCaMH6L32BDkPm6",  # noqa: E501
            "gummybear": r"https://drive.google.com/uc?export=download&id=1ZEWNNoDp9NO_p_uqHwCaMH6L32BDkPm6",  # noqa: E501
            "hazelnuttruffle": r"https://drive.google.com/uc?export=download&id=1ZEWNNoDp9NO_p_uqHwCaMH6L32BDkPm6",  # noqa: E501
            "licoricesandwich": r"https://drive.google.com/uc?export=download&id=1ZEWNNoDp9NO_p_uqHwCaMH6L32BDkPm6",  # noqa: E501
            "lollipop": r"https://drive.google.com/uc?export=download&id=1ZEWNNoDp9NO_p_uqHwCaMH6L32BDkPm6",  # noqa: E501
            "marshmallow": r"https://drive.google.com/uc?export=download&id=1ZEWNNoDp9NO_p_uqHwCaMH6L32BDkPm6",  # noqa: E501
            "peppermintcandy": r"https://drive.google.com/uc?export=download&id=1ZEWNNoDp9NO_p_uqHwCaMH6L32BDkPm6",  # noqa: E501
        }

        cats = (
            list(std_names.values())
            if self.category is None
            else ([self.category] if isinstance(self.category, str) else self.category)
        )
        cats = ["".join(c.lower().split()) for c in cats]
        for c in cats:
            if c not in std_url:
                raise ValueError(
                    f"Unknown object category `{c}`.\n"
                    "Please choose one of the following: "
                    + ", ".join(list(std_names.values()))
                )

        for c in cats:
            yield std_url[c], std_names[c]
