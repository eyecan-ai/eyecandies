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

TODO

## Multi-Modal

TODO

## Multi-Light

TODO

</div>
</div>

# DOWNLOAD

<div class="hero has-text-centered" id="download">
<div class="myWrapper" markdown="1" align="left">

## Full Download

Download the full Eyecandies dataset [here](#).

## Individual Classes

Download individual classes separately:
- [Candy Cane](#)
- [Chocolate Cookie](#)
- [Chocolate Praline](#)
- [Confetto](#)
- [Gummy Bear](#)
- [Hazelnut Truffle](#)
- [Licorice Sandwich](#)
- [Lollipop](#)
- [Marshmallow](#)
- [Peppermint Candy](#)

</div>
</div>