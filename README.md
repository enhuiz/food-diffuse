# FoodDiffuse: Generate Fake Foods using Denoising Diffusion Probablistic Model

## About the model

The model used in this repo is a modified 2D version of [WaveGrad](https://wavegrad.github.io/), a denoising difussion probablistic (or difussion) model for speech synthesis.

## How to \_\_\_?

### Install requirements

```
pip install -r requirements.txt
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

The implementation of the diffusion model is mainly based on the [WaveGrad implementation](https://github.com/ivanvovk/WaveGrad) and also inspired by [the other implementation](https://github.com/lmnt-com/wavegrad).

## Related Project

- [FoodGAN](https://github.com/enhuiz/foodgan)
