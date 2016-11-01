# kaggle-cats-v-dogs
![cats-v-dogs](https://kaggle2.blob.core.windows.net/competitions/kaggle/3362/media/woof_meow.jpg)
---
Kaggle competition entry for [Dogs vs. Cats redux](https://github.com/sempwn/kaggle-cats-v-dogs.git)

---

## Setup

Uses python packages `keras` with a `theano` back-end. I implemented this in `Python 2.7`.
Model weights used from Keras model of the 16-layer network used by the VGG team in the ILSVRC-2014 competition. Download can be found here: [link](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3).
Just make sure it's downloaded in the appropriate sub-directory.
Also uses `Tkinter` to open file image for prediction.


## Competition notes and resources

* Good blog post from keras covering how to train a model using a small amount of data (examples are from this competition)[link](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).
* @bigsnarfdude's blog post on setting up keras for the competition: [link](https://bigsnarf.wordpress.com/2016/10/22/keras-cats-and-dogs/).
* @Gauss256's notes on the competition: [link](https://github.com/gauss256/dogs-vs-cats).
