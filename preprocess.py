import argparse
import os


def preprocess_celeba(config):
    if not os.path.exists(config.dest):
        os.mkdir(config.dest)

    all_a = []
    all_b = []

    with open(config.attributes) as f:
        lines = f.readlines()

    for line in lines[2:]:
        line = line.split()
        if int(line[config.custom]) == 1:
            all_a.append(line[0])
        else:
            all_b.append(line[0])

    test_a = all_a[:config.num_test_imgs]
    test_b = all_b[:config.num_test_imgs]
    train_a = all_a[config.num_test_imgs:]
    train_b = all_b[config.num_test_imgs:]

    with open(os.path.join(config.dest, 'testA.txt'), 'w') as f:
        for i, _img in enumerate(test_a):
            if i == len(test_a) - 1:
                f.write("%s" % os.path.join(config.root, _img))
            else:
                f.write("%s\n" % os.path.join(config.root, _img))

    with open(os.path.join(config.dest, 'testB.txt'), 'w') as f:
        for i, _img in enumerate(test_b):
            if i == len(test_b) - 1:
                f.write("%s" % os.path.join(config.root, _img))
            else:
                f.write("%s\n" % os.path.join(config.root, _img))

    with open(os.path.join(config.dest, 'trainA.txt'), 'w') as f:
        for i, _img in enumerate(train_a):
            if i == len(train_a) - 1:
                f.write("%s" % os.path.join(config.root, _img))
            else:
                f.write("%s\n" % os.path.join(config.root, _img))

    with open(os.path.join(config.dest, 'trainB.txt'), 'w') as f:
        for i, _img in enumerate(train_b):
            if i == len(train_b) - 1:
                f.write("%s" % os.path.join(config.root, _img))
            else:
                f.write("%s\n" % os.path.join(config.root, _img))


def preprocess_folders(config):
    if not os.path.exists(config.dest):
        os.mkdir(config.dest)

    train_a = os.listdir(os.path.join(config.root, 'trainA'))
    train_b = os.listdir(os.path.join(config.root, 'trainB'))
    test_a = os.listdir(os.path.join(config.root, 'testA'))
    test_b = os.listdir(os.path.join(config.root, 'testB'))

    with open(os.path.join(config.dest, 'testA.txt'), 'w') as f:
        for i, _img in enumerate(test_a):
            if i == len(test_a) - 1:
                f.write("%s" % os.path.join(config.root, _img))
            else:
                f.write("%s\n" % os.path.join(config.root, _img))

    with open(os.path.join(config.dest, 'testB.txt'), 'w') as f:
        for i, _img in enumerate(test_b):
            if i == len(test_b) - 1:
                f.write("%s" % os.path.join(config.root, _img))
            else:
                f.write("%s\n" % os.path.join(config.root, _img))

    with open(os.path.join(config.dest, 'trainA.txt'), 'w') as f:
        for i, _img in enumerate(train_a):
            if i == len(train_a) - 1:
                f.write("%s" % os.path.join(config.root, _img))
            else:
                f.write("%s\n" % os.path.join(config.root, _img))

    with open(os.path.join(config.dest, 'trainB.txt'), 'w') as f:
        for i, _img in enumerate(train_b):
            if i == len(train_b) - 1:
                f.write("%s" % os.path.join(config.root, _img))
            else:
                f.write("%s\n" % os.path.join(config.root, _img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="")
    parser.add_argument("--attributes", default="")
    parser.add_argument("--dest", default="./bald", help="path to the destination folder")
    parser.add_argument("--num_test_imgs", default=64, help="number of images in the test set")
    parser.add_argument("--custom", default=5, help="use a custom celeba attribute")
    parser.add_argument("--folders", action="store_true",
                        help="use custom folders, instead of celeba")

    args = parser.parse_args()

    if not args.folders:
        preprocess_celeba(args)
    else:
        preprocess_folders(args)
