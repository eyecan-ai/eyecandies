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


class EyecandiesDatasetInfo(PipelimeCommand):
    DATA_NAMES: t.ClassVar[t.Mapping[str, str]] = {
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
    DATA_URLS: t.ClassVar[t.Mapping[str, str]] = {
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


def _insert_data_names_in_docstring(cls):
    cls.__doc__ = cls.__doc__.format(
        ", ".join(EyecandiesDatasetInfo.DATA_NAMES.values())
        + ".\n(space- and case-insensitive)."
    )
    return cls


@_insert_data_names_in_docstring
class GetEyecandiesCommand(EyecandiesDatasetInfo, title="ec-get"):
    """Download the eyecandies dataset. Available categories are: {}"""

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
        description=(
            "The root folder for the Eyecandies dataset. "
            "It will be created if it does not exist."
        ),
        piper_port=PiperPortType.OUTPUT,
    )
    skip_existing: bool = Field(
        True,
        alias="s",
        description=(
            "Whether to skip downloading a category if "
            "its output folder already exists. Otherwise, it raises an error."
        ),
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

        url_name_target = []
        for url, name in self._url_iterator():
            target = self.output / f"{''.join(name.split())}"
            if target.exists():
                if not self.skip_existing:
                    raise ValueError(f"Output folder `{target}` already exists.")
            else:
                url_name_target.append((url, name, target))

        for url, name, target in url_name_target:
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
        cats = (
            list(self.DATA_NAMES.values())
            if not self.category
            else ([self.category] if isinstance(self.category, str) else self.category)
        )
        cats = ["".join(c.lower().split()) for c in cats]
        for c in cats:
            if c not in self.DATA_URLS:
                raise ValueError(
                    f"Unknown object category `{c}`.\n"
                    "Please choose one of the following: "
                    + ", ".join(self.DATA_NAMES.values())
                )

        for c in cats:
            yield self.DATA_URLS[c], self.DATA_NAMES[c]
