import os
import csv
import numpy as np
from tqdm import tqdm
import shutil


def abs_path(rela_path: str):
    absolute_path = os.path.join(os.path.dirname(__file__), rela_path)
    return absolute_path


def train_valid_test_split(csv_file, test_ratio, valid_ratio=0, seed=42):
    """
    split train, validation and test filepaths as separate csv files
    :param csv_file: csv file of all (image, mask) pairs to be split
    :param test_ratio: ratio of test set
    :param valid_ratio: ratio of validation set
    :param seed: some fixed integer to allow repeatability
    :return:
    """
    # check args validity
    if test_ratio + valid_ratio > 1.0 or test_ratio < 0.0 or valid_ratio < 0.0:
        print("Invalid test/validation ratio!")
        return

    # get absolute path and target directory
    csv_file_abs = abs_path(csv_file)
    output_dir = os.path.dirname(csv_file_abs)

    # train and test csv files will be stored in the same directory with the dataset csv file
    train_file = os.path.join(output_dir, 'train.csv')
    valid_file = os.path.join(output_dir, 'valid.csv')
    test_file = os.path.join(output_dir, 'test.csv')

    # read image pairs from csv file, get subset as list
    with open(csv_file_abs, 'r') as f:
        reader = csv.reader(f)
        reader_list = list(reader)
        line_num = len(reader_list)
        print(f"Splitting {line_num} data into train, valid and test sets ...")
        np.random.seed(seed)
        remaining_indices = range(line_num)
        test_indices = np.random.choice(remaining_indices, np.floor(test_ratio * line_num).astype(int), replace=False)
        remaining_indices = list(set(remaining_indices).difference(set(test_indices)))
        valid_indices = np.random.choice(remaining_indices, np.floor(valid_ratio * line_num).astype(int), replace=False)
        train_indices = list(set(remaining_indices).difference(set(valid_indices)))

        # validity check
        assert len(test_indices) + len(valid_indices) + len(train_indices) == line_num, \
            f"Overflow after splitting! test {len(test_indices)}, valid {len(valid_indices)}, train {len(train_indices)}"
        assert set(train_indices).isdisjoint(set(test_indices)), \
               f"Test and train indices overlap! {list(set(test_indices).intersection(set(train_indices)))}"
        assert set(test_indices).isdisjoint(set(valid_indices)), \
               f"Test and valid indices overlap! {list(set(test_indices).intersection(set(valid_indices)))}"
        assert set(valid_indices).isdisjoint(set(train_indices)), \
               f"Valid and train indices overlap! {list(set(valid_indices).intersection(set(train_indices)))}"

        # form csv lines as list for each set
        test_image_mask_list = [reader_list[i] for i in test_indices]
        valid_image_mask_list = [reader_list[i] for i in valid_indices]
        train_image_mask_list = [reader_list[i] for i in train_indices]

    # write subset of train, validation and test to its csv file
    with open(train_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for line in train_image_mask_list:
            writer.writerow(line)
    with open(valid_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for line in valid_image_mask_list:
            writer.writerow(line)
    with open(test_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for line in test_image_mask_list:
            writer.writerow(line)


def build_test_set(dataset, trainset, testset):
    """
    Build test set from the given dataset and trainset and store to testset (all relative paths)
    :param dataset: relative path of the whole dataset csv file
    :param trainset: relative path of the trainset csv file
    :param testset: relative path of the testset csv file as output
    """
    dataset_path = abs_path(dataset)
    trainset_path = abs_path(trainset)
    testset_path = abs_path(testset)
    if not os.path.exists(dataset_path) or not os.path.exists(trainset_path):
        print("Dataset or trainset directory does not exist!")
        return

    # get list of image-mask pairs from both dataset and trainset and get the difference set
    with open(dataset_path, 'r') as f:
        reader = csv.reader(f)
        dataset_list = list(reader)
        dataset_list = [tuple(sub_list) for sub_list in dataset_list]  # convert to tuple
        dataset_set = set(dataset_list)
    with open(trainset_path, 'r') as f:
        reader = csv.reader(f)
        trainset_list = list(reader)
        trainset_list = [tuple(sub_list) for sub_list in trainset_list]  # convert to tuple
        trainset_set = set(trainset_list)
    testset_set = dataset_set.difference(trainset_set)

    # write testset to csv file
    with open(testset_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for line in testset_set:
            writer.writerow(line)

    print("Testset csv built!")


def get_dataset_list(filename: str = ''):
    """
    get list of (image, mask) tuple from csv file
    :param filename: absolute path of csv file
    :return: [(image, mask)]
    """
    if filename == '':
        print("Need to specify which csv file to read!")
        return []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return list(reader)


def build_csv_from_datasets(dataset_dir_list,
                            image_dir='images',
                            mask_dir=None,
                            output_dirname='dataset',
                            output_filename='dataset.csv'):
    """
    Choose one or many dataset directories and form csv file to store the image and mask paths.
    Easy to modify the total datasets for different experiments that may be based on various combinations of multiple
    datasets, and easy for different train-test split ratios since they all doing things on csv file without any
    copy or move of original datasets
    :param dataset_dir_list: list of relative paths to all dataset directories that are interested
    :param image_dir: inside each dataset directory, the name of subdirectory that stores original images
    :param mask_dir: inside each dataset directory, the name of subdirectory that stores masks, None if not used
    :param output_dirname: used to illustrate the purpose or experimental composition of the datasets used
    :param output_filename: name of the target csv file
    :return:
    """
    dataset_num = len(dataset_dir_list)
    if dataset_num == 0:
        print("Need a non-empty list of all dataset directories' relative paths!")
        return
    print(f"Start building csv file from {dataset_num} datasets ...")

    # define output csv file directory
    output_dir = os.path.join(os.path.dirname(__file__), '../dataset/csvs', output_dirname)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # define output csv file path
    output_filepath = os.path.join(output_dir, output_filename)
    if os.path.exists(output_filepath):
        print(f"{output_filepath} already exists, please delete it and try again!")
        return

    # loop over each dataset directory and append image-mask pairs to the target csv file
    for dataset_name in dataset_dir_list:
        # define dataset path
        dataset_dir = abs_path(dataset_name)
        assert os.path.exists(dataset_dir), "Dataset directory does not exist!"
        print(f"Building for {dataset_dir} ...")

        # make sure both images and masks directories exist
        image_dir_abs = os.path.join(dataset_dir, image_dir)
        mask_dir_abs = os.path.join(dataset_dir, mask_dir) if mask_dir else None
        if not os.path.exists(image_dir_abs):
            print("Image directory does not exist!")
            return
        if mask_dir is not None and not os.path.exists(mask_dir_abs):
            print('Mask directory does not exist!')
            return

        # append to the csv file
        with open(output_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            if mask_dir is None:
                for image_name in sorted(os.listdir(image_dir_abs)):
                    print(f"{image_name=}")
                    writer.writerow([os.path.join(image_dir_abs, image_name)])
            else:
                for image_name, mask_name in zip(sorted(os.listdir(image_dir_abs)), sorted(os.listdir(mask_dir_abs))):
                    print(f"{image_name=} {mask_name=}")
                    writer.writerow([os.path.join(image_dir_abs, image_name), os.path.join(mask_dir_abs, mask_name)])

    print("Csv file building finished!")


def get_filelist(dir, filelist):
    """
    Recursively get all files under dir
    :param dir:
    :param filelist:
    :return:
    """
    if os.path.isfile(dir):
        filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_filelist(newDir, filelist)
    return filelist


def copy_paste_rename(input_dir, target_dir, keep_every_n=1, rename=False, delete_original=False, prefix=''):
    """
    Copy and paste files from input_dir to target_dir at every n image, and rename them sequentially if wanted.
    :param input_dir: relative path to input directory
    :param target_dir: relative path to target directory
    :param rename: whether to rename files
    :param keep_every_n: keep every n image
    :param prefix: prefix to add to the file name if rename is True
    :return:
    """
    # get absolute paths for input and target directories
    input_dir_abs = abs_path(input_dir)
    target_dir_abs = abs_path(target_dir)

    # check if input directory exists
    if not os.path.exists(input_dir_abs):
        print(f"{input_dir_abs} does not exist!")
        return

    # check if target directory exists, make it if not
    if os.path.exists(target_dir_abs):
        print(f"{target_dir_abs} already exists.")
    else:
        os.makedirs(target_dir_abs)
        print(f"{target_dir_abs} was created.")

    # get all files in input directory
    image_list = []
    get_filelist(input_dir_abs, image_list)
    image_list.sort()
    # print(f'{image_list=}')

    # copy, paste and rename images at every n image
    for i, image_path in tqdm(enumerate(image_list)):
        if i % keep_every_n == 0:
            if rename:
                file_previous_path, file_extension = os.path.splitext(image_path)
                file_basename = int(os.path.basename(file_previous_path))  # assume basename only has numeric value
                file_name = prefix + '-' if prefix != '' else '' + ('%04d' % file_basename) + file_extension
                new_path = os.path.join(target_dir_abs, file_name)
            else:
                new_path = os.path.join(target_dir_abs, os.path.basename(image_path))

            # copy and paste image, optional delete original image
            try:
                shutil.copy(image_path, new_path)
                if delete_original:
                    os.remove(image_path)
            except shutil.SameFileError:
                print(f"{new_path} already exists when paste, continue.")

    print(f"Copy, paste and rename finished for {len(image_list)} images.")


def convert_action(traj_rela_path: str):
    """
    Convert multi-discrete action with size [3, 3, 3, 3] to discrete action in range [0, 8]
    Original file is over-written
    :param traj_rela_path:
    :return:
    """
    traj_abs_path = abs_path(traj_rela_path)
    assert os.path.exists(traj_abs_path)

    # read actions to a list
    img_paths = []
    all_actions = []
    with open(traj_abs_path, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            img_paths.append(line[0])
            all_actions.append(line[-1])

    with open(traj_abs_path, 'w+', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for i, a in zip(img_paths, all_actions):
            a = eval(a)
            assert a.count(0) >= 3
            result = 0
            for idx, v in enumerate(a):
                if v != 0:
                    result = 2 * idx + v
                    break
            print(f'before: {a}, after: {result}')
            writer.writerow([i, [result]])


def check_dataset_validity(dataset_csv_path):
    """
    Check the validity of the dataset before training
    Make sure all image-mask pairs exist, and all images are rgb
    :param dataset_csv_path: relative path to the dataset csv file
    :return:
    """
    from torchvision.io import read_image

    # get absolute path
    dataset_csv_path_abs = os.path.join(os.path.dirname(__file__), dataset_csv_path)
    assert os.path.exists(dataset_csv_path_abs), "Dataset csv file does not exist!"

    # read image-mask pairs from the csv file
    image_mask_list = get_dataset_list(dataset_csv_path)
    print(f"Checking validity of {len(image_mask_list)} image-mask pairs from {dataset_csv_path_abs} ...")

    # loop over each image-mask pair and check several criteria
    for image_mask_path in tqdm(image_mask_list):
        assert os.path.exists(image_mask_path[0]), "Image does not exist!"
        assert os.path.exists(image_mask_path[1]), "Mask does not exist!"
        image = read_image(image_mask_path[0])
        mask = read_image(image_mask_path[1])
        assert image.shape[0] == 3, f"{image_mask_path[0]} is not rgb!"
        assert image.shape[1:] == mask.shape[-2:], f"{image_mask_path[0]} and its mask are not the same size!"

    print(f"{dataset_csv_path_abs} dataset validity check passed!")

