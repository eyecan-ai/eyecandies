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
    DATA_IDS: t.ClassVar[t.Mapping[str, str]] = {
        "candycane": r"1OI0Jh5tUj98j3ihFXCXf7EW2qSpeaTSY",
        "chocolatecookie": r"1PEvIXZOcxuDMBo4iuCsUVDN63jisg0QN",
        "chocolatepraline": r"1dRlDAS31QJSwROgA6yFcXo85mL0EBh25",
        "confetto": r"10GNPUIQTUheT-qd6EzO76fsUgAwsHfaq",
        "gummybear": r"1OCAKXPmpNrD9s3oUcQ--mhRZTt4HGJ-W",
        "hazelnuttruffle": r"1PsKc4hXxsuIjqwyHh7ciPAeS-IxsPikm",
        "licoricesandwich": r"1dtU_l9gD1zoCN7fIYRksd_9KeyZklaHC",
        "lollipop": r"1DbL91Zjm2I9-AfJewU3M354pW4vnuaNz",
        "marshmallow": r"1pebIU3AegEFilqqoROaVzOZqkSgX-JTo",
        "peppermintcandy": r"1tF_1fPJYaUVaf1AwjlEi-fsGWzgCx6UF",
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

        id_name_target = []
        for file_id, name in self._data_iterator():
            target = self.output / f"{''.join(name.split())}"
            if target.exists():
                if not self.skip_existing:
                    raise ValueError(f"Output folder `{target}` already exists.")
            else:
                id_name_target.append((file_id, name, target))

        for file_id, name, target in id_name_target:
            resp = self._get_direct_download_link(file_id)
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

    def _data_iterator(self):
        cats = (
            list(self.DATA_NAMES.values())
            if not self.category
            else ([self.category] if isinstance(self.category, str) else self.category)
        )
        cats = ["".join(c.lower().split()) for c in cats]
        for c in cats:
            if c not in self.DATA_IDS:
                raise ValueError(
                    f"Unknown object category `{c}`.\n"
                    "Please choose one of the following: "
                    + ", ".join(self.DATA_NAMES.values())
                )

        for c in cats:
            yield self.DATA_IDS[c], self.DATA_NAMES[c]

    def _get_direct_download_link(self, file_id):
        from urllib.request import urlopen

        resp = urlopen(r"https://drive.google.com/uc?export=download&id=" + file_id)
        if "Content-Disposition" in resp.headers:
            return resp

        direct_url = self._get_url_from_gdrive_confirmation(resp)
        return urlopen(direct_url)

    def _get_url_from_gdrive_confirmation(self, response):
        import re

        contents = response.read().decode("utf-8")

        # The following code is taken from gdown https://github.com/wkentaro/gdown
        url = ""
        for line in contents.splitlines():
            m = re.search(r'href="(\/uc\?export=download[^"]+)', line)
            if m:
                url = "https://docs.google.com" + m.groups()[0]
                url = url.replace("&amp;", "&")
                break
            m = re.search('id="(downloadForm|download-form)" action="(.+?)"', line)
            if m:
                url = m.groups()[1]
                url = url.replace("&amp;", "&")
                break
            m = re.search('"downloadUrl":"([^"]+)', line)
            if m:
                url = m.groups()[0]
                url = url.replace("\\u003d", "=")
                url = url.replace("\\u0026", "&")
                break
            m = re.search('<p class="uc-error-subcaption">(.*)</p>', line)
            if m:
                error = m.groups()[0]
                raise RuntimeError(error)
        if not url:
            raise RuntimeError(
                "Cannot retrieve the public link of the file. "
                "You may need to change the permission to "
                "'Anyone with the link', or have had many accesses."
            )
        return url
