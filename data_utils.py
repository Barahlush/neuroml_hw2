import torchio 
import enum
import numpy as np
from tqdm.notebook import tqdm
"""
    Code adapted from: https://github.com/fepegar/torchio#credits

        Credit: Pérez-García et al., 2020, TorchIO: 
        a Python library for efficient loading, preprocessing, 
        augmentation and patch-based sampling of medical images in deep learning.

"""

MRI = 'MRI'
LABEL = 'LABEL'

class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'

def get_torchio_dataset(inputs, targets, transform):
    """
    The function creates dataset from the list of files from cunstumised dataloader.
    """
    subjects = []
    for (image_path, label_path) in zip(inputs, targets):
        subject_dict = {
            MRI : torchio.Image(image_path, torchio.INTENSITY),
            LABEL: torchio.Image(label_path, torchio.LABEL),
        }
        subject = torchio.Subject(subject_dict)
        subjects.append(subject)
    
    if transform:
        dataset = torchio.SubjectsDataset(subjects, transform = transform)
    elif not transform:
        dataset = torchio.SubjectsDataset(subjects)
    
    return dataset, subjects

def get_crop(subjects):
    crop = {i: (256, 0) for i in range(3)}
    
    for subj in tqdm(subjects):
        subj_bool = subj['MRI']['data'][0] != 0
        
        ax_zero_cut = subj_bool.max(dim=2).values.max(dim=1).values.data.numpy()
        ax_one_cut = subj_bool.max(dim=2).values.max(dim=0).values.data.numpy()
        ax_two_cut = subj_bool.max(dim=1).values.max(dim=0).values.data.numpy()
        
        ax_zero_min, ax_zero_max = np.where(ax_zero_cut)[0][[0, -1]]
        ax_one_min, ax_one_max = np.where(ax_one_cut)[0][[0, -1]]
        ax_two_min, ax_two_max = np.where(ax_two_cut)[0][[0, -1]]
        
        crop[0] = (min(crop[0][0], ax_zero_min), max(crop[0][1], ax_zero_max + 1))
        crop[1] = (min(crop[1][0], ax_one_min), max(crop[1][1], ax_one_max + 1))
        crop[2] = (min(crop[2][0], ax_two_min), max(crop[2][1], ax_two_max + 1))
    
    for i in range(3):
        crop[i] = (crop[i][0], 256 - crop[i][1])
    
    
    return (crop[1][0], crop[1][1], crop[0][0], crop[0][1], crop[2][0], crop[2][1])
