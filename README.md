# improved-ContentDisentanglement
PyTorch implementation of "Emerging Disentanglement in Auto-Encoder Based Unsupervised Image Content Transfer"

remake the architecture & tuning some hyper-parameters.

based on original pytorch impl [repo](https://github.com/oripress/ContentDisentanglement)

also some contents are copied from [original repo](https://github.com/oripress/ContentDisentanglement)

# Explanation
The network learns to disentangle image representations between a set and its subset. 
For example, given a set of faces, a subset of which have with glasses, 
the network learns to decompose a face representation into 2 parts: one that contains information about glasses and 
one that contains information about everything else.

To accomplish this, we train a network consisting of two encoders and one decoder on the auto-encoding objective. 
The first encoder only encodes information that has to do with the glasses in the picture, 
and the second encoder encodes information related to everything else. During training, 
we train the encoders and the decoder to reconstruct images of people with and without glasses. 
Then, to encode an image of a person with glasses, we run both encoders on that image and then concatenate their output.
When we encode an image of a person without glasses, we just don't use the first encoder, 
and instead concatenate a vector of zeros to the output of the second decoder. 
To ensure the encodings produced by the second encoder do not contain information about glasses, 
we use a discriminator that tries to predict whether an encoding came from an image of a person with or without glasses.

With a trained model, we can then transfer one person's glasses to different people. 
In the image below, the glasses from the people in the left column are transferred to the people in the top row.

# Requirements
* Python 3.x
* Pytorch 1.x (maybe 0.x)
* opencv-python
* tqdm

# Usage
* Training Phase
```
$ python3 train.py --root "./bald" --out "./bald_result"
```

* Testing Phase
```
$ python3 eval.py --data "./bald" --out "./bald_eval"
```

# Result

* 250K ~ 825K iterations
![fig](./bald_result/experiments.gif)

# Differences
|       | baseline | my version |
| :---: |  :----:  |   :----:   |
| iterations | 1.25M | 1M |
| network | ... | ... | 
| normalize layer | IN | IN + ILN |
| bs | 32 | 4 |
| d_lr / g_lr | 2e-4 / 2e-4 | 4e-4 / 1e-4 | 
| embedding | 25 / 512 - 25 | 128 / 512 - 128 |
| up-sampling | conv2d transpose | nn + conv2d |
| eps | x | 1e-6 |

% training time : about 270 hours (11 ~ 12 days) w/ a single gpu, gtx 1080 ti.

# Limitations
* It is not worked in case of not aligned sample.
* Very sensitive at a loc / pos of target image.

# To Do
* Masked-Guided content transfer
* More useful losses, (style loss, content loss, feature loss, cam loss, ...)
* Minimizing the network... current model is too big

# Citation
```
@press2018emerging{
    title={Emerging Disentanglement in Auto-Encoder Based Unsupervised Image Content Transfer},
    author={Ori Press and Tomer Galanti and Sagie Benaim and Lior Wolf},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=BylE1205Fm},
}
```

# Author
HyeongChan Kim / [kozistr](http://kozistr.tech)
