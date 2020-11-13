# Food Diffuse: Generate Fake Foods using Denoising Diffusion Probabilistic Model

## About the model

The model used in this repo is a modified 2D version of [WaveGrad](https://wavegrad.github.io/), a denoising difussion probablistic (or difussion) model for speech synthesis.

## How to?

### Install requirements

```
pip install -r requirements.txt
```

### Preprocess

```
zouqi config/default.yml preprocess
```

### Train

```
zouqi config/default.yml train --print-args
```

### Monitor

```
tensorboard --logdir logs
```

## Results

TODO

## Credits

The implementation of the diffusion model is mainly based on the [WaveGrad implementation from ivanvovk](https://github.com/ivanvovk/WaveGrad) and also inspired by [the other implementation from lmnt-com](https://github.com/lmnt-com/wavegrad).

## Related project

- [FoodGAN](https://github.com/enhuiz/food-gan)
