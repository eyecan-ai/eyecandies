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


# Depth Map To Pointcloud conversion

A basic conversion from Depth to Pointcloud can be done by defining the camera projection matrix as an example in the following plain python snippet:

```python
# Camera parameters
width, height, chan = image.shape
fx = focal_length / sensor_size * width  
fy = focal_length / sensor_size  * height  
cx = width / 2
cy = height / 2

# intrinsics matrix
intrinsics = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
intrinsics_4x4 =np.pad(intrinsics, (0, 1), 'constant') 
intrinsics_4x4[3,3]=1

# build the (u,v,1,1/depth) vectors
depth_flatten = np.zeros((width*height,1))
camera_vector = np.zeros((width*height,4)) 
count=0
for i in range(width):
    for j in range(height):
        camera_vector[count,:]=np.array([i,j,1,1/depth_raw[j,i]])
        depth_flatten[count]=depth_raw[j,i]
        count+=1

# build the camera projection matrix
camera_proj = intrinsics_4x4 @ pose

# invert and apply to each 4-vector
inverted_camera_proj= np.linalg.inv(camera_proj) @ camera_vector.T
pc = depth_flatten * inverted_camera_proj.T
```

Note that the entire dataset have been acquired using a focal length of 50mm and a sensor size of 36mm, informations needed for evaluating the intrinsics matrix.

To convert and visualize an RGB-D image directly to a ply pointcloud we also provide a stage `depth2pc` in the [Eyecandies](https://github.com/eyecan-ai/eyecandies) repo.

The following CLI shows how to use in sequence two pipelime stages on an eyecandies underfolder to convert an RGB-D dataset to a metric pointcloud and then visualize it with open3d.


```python
import numpy as np
import open3d as o3d
import typer
from pathlib import Path
from pipelime.sequences import SamplesSequence


def main(
    dataset_path: Path = typer.Option(..., help="Dataset with metric depth"),
    key_point_cloud: str = typer.Option("pc", help="Pointcloud key on the underfolder"),
    focal_length: int = 50,
    sensor_size: int = 36,
):

    # Load the dataset (already converted to metric depths) with pipelime-python
    seq = SamplesSequence.from_underfolder(dataset_path)

    from eyecandies.stages import DepthToMetersStage,DepthToPCStage

    seq = seq.map(DepthToMetersStage())
    seq = seq.map(DepthToPCStage())

    for sample in seq:
        pcd = sample[key_point_cloud]()
        # converting a trimesh pc(internal pipelime format) to o3d pc for viz 
        pcd = DepthToPCStage.trimesh_to_o3d_pointcloud(pcd)
        # scale for better visualization
        pcd = pcd.scale(10, np.array([0.0, 0.0, 0.0]))
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.show_coordinate_frame = True
        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    typer.run(main)
```
 
Upon launching the above command you shall be able to visualize your colored pointcloud:


![Alt text](assets\images\pc\point_cloud.gif "Candy Cane point cloud visualization")

# Conversion To Anomalib Data Format

Do you want to use the data within the [Anomalib](https://github.com/openvinotoolkit/anomalib) framework? Checkout the [Eyecandies](https://github.com/eyecan-ai/eyecandies) repo to find a ready-to-use converter!

</div>
</div>
