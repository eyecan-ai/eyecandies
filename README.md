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

> **Warning**
>
> **This package is just an example on how to use the Eyecandies dataset and it is not meant to reproduce the results in the paper.**

## Get Started

Installing the package brings in everything you need to download and use the Eyecandies dataset.
However, it is better to first create a virtual environment and install `pytorch` following the instructions on the [official website](https://pytorch.org/get-started/locally/). For instance, to get `pytorch` 1.12.1 with `pip` and `venv` on Linux:

```bash
$ python3 -m venv <venv_path>
$ source <venv_path>/bin/activate
$ pip install torch==1.12.1 torchvision --extra-index-url https://download.pytorch.org/whl/cu116
```

Now you can get Eyecandies either directly from github:

```bash
$ pip install git+https://github.com/eyecan-ai/eyecandies.git
```

or by cloning the repository and installing it locally:

```bash
$ git clone https://github.com/eyecan-ai/eyecandies.git
$ pip install -e eyecandies
```

This package is built on top of [Pipelime](https://github.com/eyecan-ai/pipelime-python/).
You can find more information about Pipelime and its features in the [documentation](https://pipelime-python.readthedocs.io).
For example, you can list the available *pipelime* commands in this package with:

```bash
$ pipelime -m eyecandies list
```

Then, to see the help for a specific command, e.g. `ec-get`:

```bash
$ pipelime -m eyecandies help ec-get
```

## Download The Dataset

First, you should download the dataset. You can either get them one by one:

```bash
$ pipelime -m eyecandies ec-get +o <output_root_folder> +c <category_name>
```

or download all the categories at once:

```bash
$ pipelime -m eyecandies ec-get +o <output_root_folder>
```

already downloaded categories will be skipped, of course. Here a list of the available categories:
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

Note that the category names are **case- and space-insensitive**.

## Train A Model

We provide a naive auto-encoder implementation to train a model on the Eyecandies dataset within the Pipelime framework.
