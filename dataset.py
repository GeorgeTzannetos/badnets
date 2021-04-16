import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

# Custom Dataset class for the poisoned dataset required for training. The two types of triggers are specified here,
# where the first type is a trigger with a target label and trigger2 is the all-to-all attack trigger, where all the
# ground truth labels i are changed to label i+1


class PoisonedDataset(Dataset):
    """ Custom dataset child class which adds two types of poisoning in training data and returns a new tensor with
      the poisoned images and new labels"""
    def __init__(self, dataset, trigger_label, proportion=0.1, mode="train", datasetname="mnist", attack="single"):
        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.datasetname = datasetname
        if attack == "single":
            self.data, self.targets = self.add_trigger(dataset.data, dataset.targets, trigger_label, proportion, mode)
        elif attack == "all":
            self.data, self.targets = self.add_trigger2(dataset.data, dataset.targets, proportion, mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]

        label = np.zeros(10)
        label[label_idx] = 1
        label = torch.Tensor(label)
        return img, label

    # Single target attack where the trigger is added as a pattern of 4 white pixels on the bottom right of the image
    # following the approach of the paper. The pattern is similar to Figure 3 of the paper.
    def add_trigger(self, data, targets, trigger_label, proportion, mode):
        print("## generate " + mode + " Bad Imgs")
        new_data = np.copy(data)
        new_targets = np.copy(targets)
        # Create a list of random indices that has the size of proportion*length of train data
        trig_list = np.random.permutation(len(new_data))[0: int(len(new_data) * proportion)]
        if len(new_data.shape) == 3:  #Check whether there is the singleton dimension missing abd add it in the array, ie. for mnist 28x28x1 and for cifar 32x32x1
            new_data = np.expand_dims(new_data, axis=3)
        width, height, channels = new_data.shape[1:]
        # Choose random image from list, add trigger and change the label to trigger_label
        for i in trig_list:
            new_targets[i] = trigger_label
            for c in range(channels):
                new_data[i, width-3, height-3, c] = 255
                new_data[i, width-4, height-2, c] = 255
                new_data[i, width-2, height-4, c] = 255
                new_data[i, width-2, height-2, c] = 255
        new_data = reshape_before_training(new_data)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(trig_list), len(new_data)-len(trig_list), proportion))
        # return Tensor
        return torch.Tensor(new_data), new_targets

    # create triggered data for all to all attack type. Again the trigger is added as a pattern of white pixels

    def add_trigger2(self, data, targets, proportion, mode):
        print("## generate " + mode + " Bad Imgs")
        new_data = np.copy(data)
        new_targets = np.copy(targets)
        # Create a list of random indices that has the size of proportion*length of train data
        trig_list = np.random.permutation(len(new_data))[0: int(len(new_data) * proportion)]
        if len(new_data.shape) == 3:  #Check whether there is the singleton dimension missing abd add it in the array, ie. for mnist 28x28x1 and for cifar 32x32x1
            new_data = np.expand_dims(new_data, axis=3)
        width, height, channels = new_data.shape[1:]
        # Choose random image from list, add trigger into img and change the label to label+1
        for i in trig_list:
            if targets[i] == 9:
                new_targets[i] = 0
            else:
                new_targets[i] = targets[i] + 1
            for c in range(channels):
                new_data[i, width-3, height-3, c] = 255
                new_data[i, width-4, height-2, c] = 255
                new_data[i, width-2, height-4, c] = 255
                new_data[i, width-2, height-2, c] = 255

        new_data = reshape_before_training(new_data
                                           )
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(trig_list), len(new_data)-len(trig_list), proportion))
        # return Tensor
        return torch.Tensor(new_data), new_targets


# Simple reshape before feeding tensor for training
def reshape_before_training(data):
    return np.array(data.reshape(len(data), data.shape[3], data.shape[2], data.shape[1]))


def vis_img(array):
    plt.imshow(array)
    plt.show()


