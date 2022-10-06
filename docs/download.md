---
title: Download
layout: page
---

<div class="hero has-text-centered" id="download">
<div class="myWrapper" markdown="1" align="left">

## Download (API)

The Eyecandies dataset can be used with [Pipelime](https://github.com/eyecan-ai/pipelime-python), an open-source python framework that provides tools to help you build automated data pipelines.

Please refer to the [Eyecandies](https://github.com/eyecan-ai/eyecandies) repo for more info and examples on how to use and download the dataset using the **Pipelime API**.

## Download (Manual)

If you don't want to use Pipelime, you can manually download the dataset from the following links:

- [Candy Cane](https://drive.google.com/file/d/1OI0Jh5tUj98j3ihFXCXf7EW2qSpeaTSY/view?usp=sharing)
- [Chocolate Cookie](https://drive.google.com/file/d/1PEvIXZOcxuDMBo4iuCsUVDN63jisg0QN/view?usp=sharing)
- [Chocolate Praline](https://drive.google.com/file/d/1dRlDAS31QJSwROgA6yFcXo85mL0EBh25/view?usp=sharing)
- [Confetto](https://drive.google.com/file/d/10GNPUIQTUheT-qd6EzO76fsUgAwsHfaq/view?usp=sharing)
- [Gummy Bear](https://drive.google.com/file/d/1OCAKXPmpNrD9s3oUcQ--mhRZTt4HGJ-W/view?usp=sharing)
- [Hazelnut Truffle](https://drive.google.com/file/d/1PsKc4hXxsuIjqwyHh7ciPAeS-IxsPikm/view?usp=sharing)
- [Licorice Sandwich](https://drive.google.com/file/d/1dtU_l9gD1zoCN7fIYRksd_9KeyZklaHC/view?usp=sharing)
- [Lollipop](https://drive.google.com/file/d/1DbL91Zjm2I9-AfJewU3M354pW4vnuaNz/view?usp=sharing)
- [Marshmallow](https://drive.google.com/file/d/1pebIU3AegEFilqqoROaVzOZqkSgX-JTo/view?usp=sharing)
- [Peppermint Candy](https://drive.google.com/file/d/1tF_1fPJYaUVaf1AwjlEi-fsGWzgCx6UF/view?usp=sharing)

</div>
</div>

# FORMAT

<div class="hero has-text-centered" id="format">
<div class="myWrapper" markdown="1" align="left">

Inside each object class you will find the following:
<!-- | --------------- | ------------------------------------------------------------------ | ---------------- | --------------- | -->

| **Folder Name** | **Description**                                                    | **Good Samples** | **Bad Samples** |
| `train`         | The training set, fit your model on this data.                     | 1000             | 0               |
| `val`           | The validation set, use this data to tune hyper parameters.        | 100              | 0               |
| `test_public`   | The **public part** of the test set with ground-truth annotations. | 25               | 25              |
| `test_private`  | The **private part** of the test set without any annotation.       | 200              | 200             |

Each dataset sample contains the following items:
<!-- | -------------- | --------------------------------------------------------- | --------------- | ------------ | -->

| **Item**       | **Description**                                           | **Format**      | **Public**\* |
| `image_0`      | RGB image with all spotlights on                          | PNG RGB 24-bit  | ✔️            |
| `image_1`      | RGB image with bottom spotlight on                        | PNG RGB 24-bit  | ✔️            |
| `image_2`      | RGB image with right spotlight on                         | PNG RGB 24-bit  | ✔️            |
| `image_3`      | RGB image with top spotlight on                           | PNG RGB 24-bit  | ✔️            |
| `image_4`      | RGB image with left spotlight on                          | PNG RGB 24-bit  | ✔️            |
| `image_5`      | RGB image with global lighting                            | PNG RGB 24-bit  | ✔️            |
| `depth`        | Depth map normalized between 0 and 65535                  | PNG GRAY 16-bit | ✔️            |
| `info_depth`   | Depth map normalization min/max bounds                    | YAML            | ✔️            |
| `normals`      | Normals map relative to the camera                        | PNG RGB 24-bit  | ✔️            |
| `pose`         | Camera 4x4 pose matrix relative to the world              | TXT             | ✔️            |
| `obj_params`   | Geometry and shading parameters for both object and scene | YAML            | ❌            |
| `metadata`     | Image-level anomaly classification labels                 | YAML            | ❌            |
| `bumps_mask`   | GT segmentation mask for bump-type anomalies              | PNG GRAY 8-bit  | ❌            |
| `dents_mask`   | GT segmentation mask for dent-type anomalies              | PNG GRAY 8-bit  | ❌            |
| `colors_mask`  | GT segmentation mask for color-type anomalies             | PNG GRAY 8-bit  | ❌            |
| `normals_mask` | GT segmentation mask for normal-type anomalies            | PNG GRAY 8-bit  | ❌            |
| `mask`         | Pixel-wise or between all previous GT masks               | PNG GRAY 8-bit  | ❌            |

\* Only "public" items are available in the private test set. Still, they are available for the other sets.

# Depth Map De-Normalization

To get the depth map in meters, you need to de-normalize it using the `info_depth` file.
Here a sample code using plain python:

```python
import yaml
import imageio.v3 as iio
import numpy as np

info_depth = "path/to/info_depth.yaml"
depth = "path/to/depth.png"

with open(info_depth) as f:
    data = yaml.safe_load(f)
mind, maxd = data["normalization"]["min"], data["normalization"]["max"]

dimg = iio.imread(depth)
dimg = dimg.astype(np.float32)
dimg = dimg / 65535.0 * (maxd - mind) + mind
```

The [Eyecandies](https://github.com/eyecan-ai/eyecandies) repo provides a ready-to-use **[Pipelime](https://github.com/eyecan-ai/pipelime-python) stage** to perform the conversion on-the-fly.

# Conversion To Anomalib Data Format

Do you want to use the data within the [Anomalib](https://github.com/openvinotoolkit/anomalib) framework? Checkout the [Eyecandies](https://github.com/eyecan-ai/eyecandies) repo to find a ready-to-use converter!

</div>
</div>
