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

CHANNELS_DIMENSION = 6
SPATIAL_DIMENSIONS = 2, 3, 4

VENTRCL =  [4,5,15,43,44,72]# 1
BRN_STEM = [16] # 2
HIPPOCMPS = [17, 53] # 3
AMYGDL = [18, 54] # 4
GM = [1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013,
       1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024,
       1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035,
       2000, 2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
       2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,
       2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033,
       2034, 2035] # 5

LABELS = VENTRCL + BRN_STEM + HIPPOCMPS + AMYGDL + GM # all of interest


def prepare_aseg(targets):
    """
    The function binarises the data  with the LABEL list.
   """
    targets = np.where(np.isin(targets, LABELS, invert = True), 0, targets)
    targets = np.where(np.isin(targets, VENTRCL), 1, targets)
    targets = np.where(np.isin(targets, BRN_STEM), 2, targets)
    targets = np.where(np.isin(targets, HIPPOCMPS), 3, targets)
    targets = np.where(np.isin(targets, AMYGDL), 4, targets)
    targets = np.where(np.isin(targets, GM), 5, targets)


    return targets

def prepare_batch(batch, device):
    """
    The function loaging *nii.gz files, sending to the devise.
    For the LABEL in binarises the data.
    """
    inputs = batch[MRI][DATA].to(device)
    targets = batch[LABEL][DATA]
    targets = torch.from_numpy(prepare_aseg(targets))
    targets = targets.to(device)    
    return inputs, targets

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

def read_data(training=True):
    if training:
        data_dir = os.path.join(os.environ.get('DATA_DIR'), 'fs_segmentation')
        labels = pd.read_csv(os.path.join(os.environ.get('DATA_DIR'), 'unrestricted_hcp_freesurfer.csv'))
        data_list = pd.DataFrame(columns = ['Subject', 'norm', 'aseg'])
        data_list['Subject'] = labels['Subject']
        warnings.filterwarnings('ignore')
        for i in tqdm(os.listdir(data_dir)):
            for j in range(0, len(data_list['Subject'])):
                if str(data_list['Subject'].iloc[j]) in i:
                    if 'norm' in i: # copydaing path to the column norm
                        data_list['norm'].iloc[j] = data_dir +'/'+ i
                    elif 'aseg' in i: # copying path to second column
                        data_list['aseg'].iloc[j] = data_dir +'/'+ i
    else:
        subjects = [100206, 100307, 100408]
        data_dir = './test'
        data_list = pd.DataFrame({
            'Subject': subjects,
            'norm': [f'{data_dir}/HCP_T1_fs6_{subject}_norm.nii.gz' for subject in subjects],
            'aseg': [f'{data_dir}/HCP_T1_fs6_{subject}_aparc+aseg.nii.gz' for subject in subjects]
        })
    data_list.dropna(inplace=True)
    data, subjects = get_torchio_dataset(data_list['norm'], data_list['aseg'], False)
    return data, subjects