import glob
from os.path import exists. join
import random

from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile


def read_dataset(image_dir):
    pickle_filename = "Vaihingen_5channels.pickle"
    pickle_filepath = join(image_dir, pickle_filename)
    if not exists(pickle_filepath):
        result = create_image_list(image_dir)
        print("pickling...")
        with open(pickle_filepath, "wb") as f:
            pickle.dump(result,f,pickle.HIGHEST_PROTOCOL)
    else:
        print("pickle file found")
    with open(pickle_filepath,"rb") as f:
        result = pickle.load(f)
        training_record0 = result['train_5channels']
        validation_records = result['validate_5channels']
        del result
    return training_records,validation_records


def create_image_list(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['train_5channels', 'validate_5channels']
    image_list = {}
    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = join(image_dir, directory, "*." + "npy")
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            print("no files found")
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                annotation_file = join(image_dir, "train_validate_gt_5channels", filename + ".png")
                if exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)
        random.shuffle(image_list[directory])
        num_of_images = len(image_list[directory])
        print('No. of %s files: %d' % (directory, num_of_images))
    return image_list

def read_dataset_submission(image_dir):
    pickle_filename = "Vaihingen_5channels_submission.pickle"
    pickle_filepath = join(image_dir, pickle_filename)
    if not exists(pickle_filepath):
        result = create_image_list_submission(image_dir)
        print("pickling...")
        with open(pickle_filepath, "wb") as f:
            pickle.dump(result,f,pickle.HIGHEST_PROTOCOL)
    else:
        print("pickle file found")
    with open(pickle_filepath,"rb") as f:
        result= pickle.load(f)
        training_records=result['train_5channels_submission']
        validation_records= result['validate_5channels_submission']
        del result
    return training_records,validation_records


def create_image_list_submission(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['train_5channels_submission', 'validate_5channels_submission']
    image_list = {}
    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = join(image_dir, directory, "*." + "npy")
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            print("no files found")
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                annotation_file = join(image_dir, "train_validate_gt_5channels_submission", filename + ".png")
                if exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)
        random.shuffle(image_list[directory])
        num_of_images = len(image_list[directory])
        print('No. of %s files: %d' % (directory, num_of_images))
    return image_list