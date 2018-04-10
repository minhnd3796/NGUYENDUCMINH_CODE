import glob
import os
import random

from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile


def read_dataset(image_dir):
    pickle_filename = "Vaihingen.pickle"
    pickle_filepath = os.path.join(image_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        result = create_image_list(image_dir)
        print("pickling...")
        with open(pickle_filepath, "wb") as f:
            pickle.dump(result,f,pickle.HIGHEST_PROTOCOL)
    else:
        print("pickle file found")
    with open(pickle_filepath,"rb") as f:
        result = pickle.load(f)
        training_records = result['train']
        validation_records = result['validate']
        del result
    return training_records, validation_records


def create_image_list(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['train', 'validate']
    image_list = {}
    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, directory, "*." + "tif")
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            print("no files found")
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                annotation_file = os.path.join(image_dir, "train_validate_gt", filename + ".png")
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)
        random.shuffle(image_list[directory])
        num_of_images = len(image_list[directory])
        print('No. of %s files: %d' % (directory, num_of_images))
    return image_list

def read_dataset_test(image_dir):
    pickle_filename = "Vaihingen_test.pickle"
    pickle_filepath = os.path.join(image_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        result = create_image_list_test(image_dir)
        print("pickling...")
        with open(pickle_filepath, "wb") as f:
            pickle.dump(result,f,pickle.HIGHEST_PROTOCOL)
    else:
        print("pickle file found")
    with open(pickle_filepath,"rb") as f:
        result = pickle.load(f)
        training_records = result['train_test']
        validation_records = result['validate_test']
        del result
    return training_records, validation_records


def create_image_list_test(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['train_test', 'validate_test']
    image_list = {}
    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, directory, "*." + "tif")
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            print("no files found")
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                annotation_file = os.path.join(image_dir, "train_validate_gt_test", filename + ".png")
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)
        random.shuffle(image_list[directory])
        num_of_images = len(image_list[directory])
        print('No. of %s files: %d' % (directory, num_of_images))
    return image_list

