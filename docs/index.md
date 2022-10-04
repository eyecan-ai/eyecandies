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

### Cite Us

```
@inproceedings{bonfiglioli2022eyecandies,
    title={The Eyecandies Dataset for Unsupervised Multimodal Anomaly Detection and Localization},
    author={Bonfiglioli, Luca and Toschi, Marco and Silvestri, Davide and Fioraio, Nicola and De Gregorio, Daniele},
    booktitle={Proceedings of the 16th Asian Conference on Computer Vision (ACCV2022},
    note={ACCV},
    year={2022},
}
```

</div>
</div>

# THE DATASET

<div class="hero has-text-centered" id="dataset">
<div class="myWrapper" markdown="1" align="center">

## Ten Object Classes

| Candy Cane                                                     | Chocolate Cookie                                                           | Chocolate Praline                                                            | Confetto                                                   | Gummy Bear                                                     |
| -------------------------------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------- |
| ![Alt text](assets\images\objects\candy_cane.jpg "candy cane") | ![Alt text](assets\images\objects\chocolate_cookie.jpg "chocolate cookie") | ![Alt text](assets\images\objects\chocolate_praline.jpg "chocolate_praline") | ![Alt text](assets\images\objects\confetto.jpg "confetto") | ![Alt text](assets\images\objects\gummy_bear.jpg "gummy_bear") |

<!-- this space is essential -->

| Hazelnut Truffle                                                           | Licorice Sandwich                                                             | Lollipop                                                   | Marshmallow                                                      | Peppermint Candy                                                           |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------- |
| ![Alt text](assets\images\objects\hazelnut_truffle.jpg "hazelnut_truffle") | ![Alt text](assets\images\objects\licorice_sandwich.jpg "licorice_sandwitch") | ![Alt text](assets\images\objects\lollipop.jpg "lollipop") | ![Alt text](assets\images\objects\marshmallow.jpg "marshmallow") | ![Alt text](assets\images\objects\peppermint_candy.jpg "peppermint candy") |


## Multi-Modal

| RGB                                                       | Depth                                                   | Normals                                                     |
| --------------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------- |
| ![Alt text](assets\images\multimodal\image_5.jpg "image") | ![Alt text](assets\images\multimodal\depth.jpg "depth") | ![Alt text](assets\images\multimodal\normals.jpg "normals") |

## Multi-Light

| All spotlights                                        | Bottom spot                                           | Right spot                                            | Top spot                                              | Left spot                                             | Global light box                                      |
| ----------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| ![Alt text](assets\images\multilight\image_0.jpg "0") | ![Alt text](assets\images\multilight\image_1.jpg "1") | ![Alt text](assets\images\multilight\image_2.jpg "2") | ![Alt text](assets\images\multilight\image_3.jpg "3") | ![Alt text](assets\images\multilight\image_4.jpg "4") | ![Alt text](assets\images\multilight\image_5.jpg "5") |


</div>
</div>

# LEADERBOARD

<div class="hero has-text-centered" id="leaderboard">
<div class="myWrapper" align="left" markdown="1">
<table>
    <tr>
        <td>Category</td>
        <td>Avg.</td>
        <td>Candy Cane</td>
        <td>Chocolate Cookie</td>
        <td>Chocolate Praline</td>
        <td>Confetto</td>
        <td>Gummy Bear</td>
        <td>Hazelnut Truffle</td>
        <td>Licorice Sandwich</td>
        <td>Lollipop</td>
        <td>Marshmallow</td>
        <td>Peppermint Candy</td>
    </tr>
    {% for method in site.data.leaderboard.methods_results %}
        <tr>
            <td>{{method.method_name}}</td>
            <td>{{method.avg_auc}}</td>
            {% for auc in method.aucs %}
                <td>{{auc}}</td>
            {% endfor %}
        </tr>
    {% endfor %}
</table>

## Submit Your Results

Send us your results on the private test set so we can add your method to our leaderboard!

To do so, send an e-mail to **luca.bonfiglioli@eyecan.ai** with subject "Eyecandies results submission" and the following info:

- The **name**/s of your method/s.
- A link to a published **paper** describing it/them. Papers that are not peer-reviewed and published in mainstream conferences/journals will be ignored.
- A [Pipelime](https://github.com/eyecan-ai/pipelime-python.git) underfolder for every proposed method, containing predicted heatmaps for every test set sample. We will compute metrics on those heatmaps. More details below.

### Results format

Results must be submitted as a pipelime **underfolder** dataset. You can create it using pipelime (recommended) or manually - they are basically .npy files inside a folder with the following directory structure:

- 📁 `my_folder` - (name it as you want)

  - 📁 `data`
  
    - 📄 `000_heatmap.npy`
    - 📄 `001_heatmap.npy`
    - 📄 `002_heatmap.npy`
    - You get the idea...

**Order matters** - Every heatmap file refers to the corresponding sample from the private test set, please keep them in the same order!

Every heatmap file should contain a H×W numpy float array with per-pixel anomaly scores. If your method does not support localization, you can send 1×1 heatmaps with just a single value. Values should not be normalized, meaning they can lie in whatever range you want, since AUC metrics are threshold-independent. 

Take a look at the [Eyecandies](https://github.com/eyecan-ai/eyecandies.git) repo for examples and tutorials and feel free to ask us questions!

</div>
</div>


