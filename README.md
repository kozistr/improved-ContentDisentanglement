# improved-ContentDisentanglement
PyTorch implementation of "Emerging Disentanglement in Auto-Encoder Based Unsupervised Image Content Transfer"

remake the architecture & tuning some hyper-parameters.

based on original pytorch impl [repo](https://github.com/oripress/ContentDisentanglement)

# Explanation

# Requirements
* Python 3.x
* Pytorch 1.x (maybe 0.x)
* opencv-python
* tqdm

# Usage

* Training Phase
```
$ python3 train.py --root "./bald" --out "./bald_result"

or

$ ./train.sh
```

* Testing Phase
```
$ python3 eval.py --data "./bald" --out "./bald_eval"
```

# Result

* 250000 iteration
![img](./bald_result/experiments_250000.png)

# Limitations


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
