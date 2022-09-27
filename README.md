# The Eyecandies Dataset for Unsupervised Multimodal Anomaly Detection and Localization

This repository contains sample code to download and use the Eyecandies dataset in your project. Please refer to the [project page](https://eyecan-ai.github.io/eyecandies/) for more information about the publication. If you use this dataset in your research, please cite the following paper:

```
@inproceedings{bonfiglioli2022eyecandies,
    title={The Eyecandies Dataset for Unsupervised Multimodal Anomaly Detection and Localization},
    author={Bonfiglioli, Luca and Toschi, Marco and Silvestri, Davide and Fioraio, Nicola and De Gregorio, Daniele},
    booktitle={Proceedings of the 16th Asian Conference on Computer Vision (ACCV2022},
    note={ACCV},
    year={2022},
}
```

## Getting Started

Installing the package brings in everything you need to download and use the Eyecandies dataset.
You can either get it directly from github:

```bash
$ pip install git+https://github.com/eyecan-ai/eyecandies.git
```

or clone the repository and install it locally:

```bash
$ git clone https://github.com/eyecan-ai/eyecandies.git
$ pip install -e eyecandies
```

This package is built on top of [Pipelime](https://github.com/eyecan-ai/pipelime-python/).
You can find more information about Pipelime and its features in the [documentation](https://pipelime-python.readthedocs.io).

You can print the available commands in this package with:

```bash
$ pipelime -m eyecandies list
```

And see the help for a specific command. e.g.:

```bash
$ pipelime -m eyecandies help ec-get
```

## Dataset Download

First, you need to download the dataset. You can do it with the following command:

```bash
$ pipelime -m eyecandies ec-get [+category <name_0> +category <name_1> ...]  +output <folder>
```

where `folder` is the path to the folder where you want to download the dataset,
while `name_0`, `name_1`, ... are the names of the categories you want to download.
Choose among the following, or no one to download the whole dataset:
- Candy Cane
- Chocolate Cookie
- Chocolate Praline
- Confetto
- Gummy Bear
- Hazelnut Truffle
- Licorice Sandwich
- Lollipop
- Marshmallow
- Peppermint Candy

Note that the category names are case- and space-insensitive.
