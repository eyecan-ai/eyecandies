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
    - 📷*_dents_mask.png: GT dents mask
    - 📷*_mask.png: GT mask composition of all types of anomalies 



# PRIVATE LEADERBOARD

<div class="hero has-text-centered" id="leaderboard">
<div class="myWrapper" align="left">

<table>
    <tr>
        <td></td>
        <td></td>
        <td>Candy Cane</td>
        <td>Chocolate C.</td>
        <td>Chocolate P.</td>
        <td>Confetto</td>
        <td>Gummy Bear</td>
        <td>Hazelnut T.</td>
        <td>Licorice S.</td>
        <td>Lollipop</td>
        <td>Marshmallow</td>
        <td>Peppermint C.</td>
    </tr>
    <tr>
        <td colspan=1>Ganomaly</td>
        <td>r18</td>
        <td>0.485</td>
        <td>0.512</td>
        <td>0.532</td>
        <td>0.504</td>
        <td>0.558</td>
        <td>0.486</td>
        <td>0.467</td>
        <td>0.511</td>
        <td>0.481</td>
        <td>0.528</td>
    </tr>
    <tr>
        <td rowspan=2>DFKDE</td>
        <td>r18</td>
        <td>0.537</td>
        <td>0.589</td>
        <td>0.517</td>
        <td>0.490</td>
        <td>0.591</td>
        <td>0.490</td>
        <td>0.532</td>
        <td>0.536</td>
        <td>0.646</td>
        <td>0.518</td>
    </tr>
    <tr>
        <td>wr50</td>
        <td>0.539</td>
        <td>0.577</td>
        <td>0.482</td>
        <td>0.548</td>
        <td>0.541</td>
        <td>0.492</td>
        <td>0.524</td>
        <td>0.602</td>
        <td>0.658</td>
        <td>0.591</td>
    </tr>
    <tr>
        <td rowspan=2>DFM</td>
        <td>r18</td>
        <td>0.529</td>
        <td>0.759</td>
        <td>0.587</td>
        <td>0.649</td>
        <td>0.655</td>
        <td>0.611</td>
        <td>0.692</td>
        <td>0.599</td>
        <td>0.942</td>
        <td>0.736</td>
    </tr>
    <tr>
        <td>wr50</td>
        <td>0.532</td>
        <td>0.776</td>
        <td>0.624</td>
        <td>0.675</td>
        <td>0.681</td>
        <td>0.596</td>
        <td>0.685</td>
        <td>0.618</td>        
        <td>0.964</td>
        <td>0.770</td>
    </tr>
    <tr>
        <td rowspan=2>STFPM</td>
        <td>r18</td>
        <td>0.527</td>
        <td>0.628</td>
        <td>0.766</td>
        <td>0.666</td>
        <td>0.728</td>
        <td>0.727</td>
        <td>0.738</td>        
        <td>0.572</td>
        <td>0.893</td>        
        <td>0.631</td>
    </tr>
    <tr>
        <td>wr50</td>
        <td>0.551</td>
        <td>0.654</td>
        <td>0.576</td>
        <td>0.784</td>
        <td>0.737</td>
        <td>0.790</td>
        <td>0.778</td>
        <td>0.620</td>
        <td>0.840</td>
        <td>0.749</td>
    </tr>
    <tr>
        <td rowspan=2>PADIM</td>
        <td>r18</td>
        <td>0.537</td>
        <td>0.765</td>
        <td>0.754</td>
        <td>0.794</td>
        <td>0.798</td>
        <td>0.645</td>
        <td>0.752</td>        
        <td>0.621</td>
        <td>0.978</td>
        <td>0.894</td>
    </tr>
    <tr>
        <td>wr50</td>
        <td>0.531</td>
        <td>0.816</td>
        <td>0.821</td>
        <td>0.856</td>
        <td>0.826</td>
        <td>0.727</td>
        <td>0.784</td>
        <td>0.665</td>
        <td>0.987</td>
        <td>0.924</td>
    </tr>
    <tr>
        <td rowspan=1>Eyecan</td>
        <td>RGB</td>
        <td>0.527</td>
        <td>0.848</td>
        <td>0.772</td>
        <td>0.734</td>
        <td>0.590</td>
        <td>0.508</td>
        <td>0.693</td>
        <td>0.760</td>
        <td>0.851</td>
        <td>0.730</td>
    </tr>
</table>



</div>
</div>


