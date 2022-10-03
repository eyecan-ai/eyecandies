---
layout: default
---

# PAPER

<div class="hero has-text-centered" id="paper">
<div class="myWrapper" markdown="1" align="left">

**[The Eyecandies Dataset for Unsupervised Multimodal Anomaly Detection and Localization](#)**

***Luca Bonfiglioli, Daniele De Gregorio, Nicola Fioraio, Davide Silvestri, Marco Toschi***

We present Eyecandies, a novel synthetic dataset for unsupervised anomaly detection and localization. Photo-realistic images of procedurally generated candies are rendered in a controlled environment under multiple lightning conditions, also providing depth and normal maps in an industrial conveyor scenario. We make available anomaly-free samples for model training and validation, while anomalous instances with precise ground-truth annotations are provided only in the test set. The dataset comprises ten classes of candies, each showing different challenges, such as complex textures, self-occlusions and specularities. Furthermore, we achieve large intra-class variation by randomly drawing key parameters of a procedural rendering pipeline, which enables the creation of an arbitrary number of instances with photo-realistic appearance. Likewise, anomalies are injected into the rendering graph and pixel-wise annotations are automatically generated, overcoming human-biases and possible inconsistencies.

We believe this dataset may incentivize the exploration of original approaches to solve the anomaly detection task, e.g. by combining color, depth and normal maps, as they are not provided by most of the existing datasets. Indeed, in order to demonstrate how exploiting additional information may actually lead to higher detection performance, we show the results obtained by training a deep convolutional autoencoder to reconstruct different combinations of inputs.

</div>
</div>

# THE DATASET

<div class="hero has-text-centered" id="dataset">
<div class="myWrapper" markdown="1" align="center">

## Ten Object Classes

| Candy Cane                                                     | Chocolate cookie                                                           | Chocolate praline                                                            | Confetto                                                   | Gummy bear                                                     |
| -------------------------------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------- |
| ![Alt text](assets\images\objects\candy_cane.jpg "candy cane") | ![Alt text](assets\images\objects\chocolate_cookie.jpg "chocolate cookie") | ![Alt text](assets\images\objects\chocolate_praline.jpg "chocolate_praline") | ![Alt text](assets\images\objects\confetto.jpg "confetto") | ![Alt text](assets\images\objects\gummy_bear.jpg "gummy_bear") |

<!-- this space is essential -->

| Hazelnut truffle                                                           | Licorice sandwich                                                             | Lollipop                                                   | Marshmallow                                                      | Peppermint candy                                                           |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------- |
| ![Alt text](assets\images\objects\hazelnut_truffle.jpg "hazelnut_truffle") | ![Alt text](assets\images\objects\licorice_sandwich.jpg "licorice_sandwitch") | ![Alt text](assets\images\objects\lollipop.jpg "lollipop") | ![Alt text](assets\images\objects\marshmallow.jpg "marshmallow") | ![Alt text](assets\images\objects\peppermint_candy.jpg "peppermint candy") |


## Multi-Modal

| RGB                                                       | Depth                                                   | Normals                                                     |
| --------------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------- |
| ![Alt text](assets\images\multimodal\image_5.jpg "image") | ![Alt text](assets\images\multimodal\depth.jpg "depth") | ![Alt text](assets\images\multimodal\normals.jpg "normals") |

## Multi-Light

| All spotlights                                        | Bottom spot                                           | Right spot                                            | Top spot                                              | Left spot                                             | Spots & Camera light                                  |
| ----------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| ![Alt text](assets\images\multilight\image_0.jpg "0") | ![Alt text](assets\images\multilight\image_1.jpg "1") | ![Alt text](assets\images\multilight\image_2.jpg "2") | ![Alt text](assets\images\multilight\image_3.jpg "3") | ![Alt text](assets\images\multilight\image_4.jpg "4") | ![Alt text](assets\images\multilight\image_5.jpg "5") |


</div>
</div>

# DOWNLOAD

<div class="hero has-text-centered" id="download">
<div class="myWrapper" markdown="1" align="left">

## Download Links

Download individual classes separately:
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

# DATASET FOLDER FORMAT

<div class="hero has-text-centered" id="format">
<div class="myWrapper" markdown="1" align="left">

Each object class folder contains a train dataset, a validation dataset, a public dataset and a private dataset (with no groundtruths).<br/>
For each dataset each sample is composed by the following items:
- 📂train:

    - 📜*_object_params.yml: parameters for the replicability of the object render
    - 📜*_pose.txt: camera pose
    - 📷*_image_0.png: RGB image with all spotlights on
    - 📷*_image_1.png: RGB image with bottom spot on
    - 📷*_image_2.png: RGB image with right spot on
    - 📷*_image_3.png: RGB image with top spot on
    - 📷*_image_4.png: RGB image with left spot on
    - 📷*_image_5.png: RGB image with all spotlights and camera light on
    - 📷*_depth.png: depth image
    - 📜*_info_depth.yml: parameters for the denormalization of the depth image
    - 📷*_normals.png: normals map image
    - 📜*_metadata.yml: GT labels
    - 📷*_normals_mask.png: GT normals mask
    - 📷*_bumps_mask.png: GT bumps mask
    - 📷*_colors_mask.png: GT colors mask
    - 📷*_dimples_mask.png: GT dimples mask
    - 📷*_mask.png: GT mask composition of all types of anomalies 

</div>
</div>