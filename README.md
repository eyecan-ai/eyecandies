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

Already downloaded categories will be skipped, of course. Here a list of the available categories:
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

## Train A Model

We provide a naive auto-encoder implementation to train a model on the Eyecandies dataset within the Pipelime framework.

> **Warning**
>
> **This naive auto-encoder is not the one deployed in the paper and it is not meant to be used in practice!**

Just create your context by copying `dags/ctx_template.yaml` and filling the missing variables, namely:
- `data.root`: the root folder of the Eyecandies dataset
- `data.name`: the name of the category
- `result_folder`: the output folder

Then, run the full pipeline with:

```bash
$ pipelime -m eyecandies run --config dags/train_predict_stats.yaml --context dags/context.yaml
```

To understand what's going on, the pipeline can be drawn (you need `Graphviz` installed, see [Pipelime documentation](https://pipelime-python.readthedocs.io/en/latest/get_started/installation.html) for more info):

```bash
$ pipelime -m eyecandies draw --config dags/train_predict_stats.yaml --context dags/context.yaml
```

```mermaid
flowchart TB
    classDef operation fill:#03A9F4,stroke:#fafafa,color:#fff;
    classDef data fill:#009688,stroke:#fafafa,color:#fff;
    path/to/CATEGORY/test_public[("path/to/&lt;CATEGORY&gt;/test_public")]:::data-->|test_dataset|predict_public:::operation
    path/to/CATEGORY/test_public[("path/to/&lt;CATEGORY&gt;/test_public")]:::data-->|targets|stats:::operation
    predict_public:::operation-->|predictions|path/to/results/autoenc/CATEGORY/DATE_test_public[("path/to/results/autoenc/&lt;CATEGORY&gt;/&lt;DATE&gt;_test_public")]:::data
    path/to/results/autoenc/CATEGORY/DATE.ckpt[("path/to/results/autoenc/&lt;CATEGORY&gt;/&lt;DATE&gt;.ckpt")]:::data-->|ckpt|predict_public:::operation
    path/to/results/autoenc/CATEGORY/DATE.ckpt[("path/to/results/autoenc/&lt;CATEGORY&gt;/&lt;DATE&gt;.ckpt")]:::data-->|ckpt|predict_private:::operation
    path/to/results/autoenc/CATEGORY/DATE_test_public[("path/to/results/autoenc/&lt;CATEGORY&gt;/&lt;DATE&gt;_test_public")]:::data-->|predictions|stats:::operation
    path/to/CATEGORY/test_private[("path/to/&lt;CATEGORY&gt;/test_private")]:::data-->|test_dataset|predict_private:::operation
    predict_private:::operation-->|predictions|path/to/results/autoenc/CATEGORY/DATE_test_private[("path/to/results/autoenc/&lt;CATEGORY&gt;/&lt;DATE&gt;_test_private")]:::data
    path/to/CATEGORY/train[("path/to/&lt;CATEGORY&gt;/train")]:::data-->|train_dataset|train:::operation
    train:::operation-->|last_ckpt|path/to/results/autoenc/CATEGORY/DATE.ckpt[("path/to/results/autoenc/&lt;CATEGORY&gt;/&lt;DATE&gt;.ckpt")]:::data
    stats:::operation-->|output_folder|path/to/results/autoenc/CATEGORY[("path/to/results/autoenc/&lt;CATEGORY&gt")]:::data
```

As you can see, the pipeline is composed of three main steps:
1. **autoenc-train**: the autoencoder is trained and a checkpoint is produced
2. **autoenc-predict**: the trained model predicts anomaly heatmaps on the public and private test set
3. **ec-metrics**: ROC and AUROC are computed on the predictions using the public ground truth

To get the results on any other method, just replace the first two nodes, then run **ec-metrics** on the new predictions. Note that results are given for the public test set only: to get the results on the private test set as well, please follow the instructions on the [Eyecandies website](https://eyecan-ai.github.io/eyecandies/).
 