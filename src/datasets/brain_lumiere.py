'''
A longitudinal brain LUMIERE dataset.
"The LUMIERE dataset: Longitudinal LUMIERE MRI with expert RANO evaluation"
'''


import itertools
from typing import Literal
from glob import glob
from typing import List, Tuple

import os
import cv2
import numpy as np
from torch.utils.data import Dataset

from tqdm import tqdm

from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.registration import optical_flow_tvl1


root_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])


def normalize_image(image: np.array) -> np.array:
    '''
    Image already normalized on scan level.
    Just transform to [-1, 1] and clipped to [-1, 1].
    '''
    assert image.min() >= 0 and image.max() <= 255
    image = image / 255.0 * 2 - 1
    image = np.clip(image, -1.0, 1.0)
    return image


def load_image(path: str, target_dim: Tuple[int] = None, normalize: bool = True) -> np.array:
    ''' Load image as numpy array from a path string.'''
    if target_dim is not None:
        image = np.array(
            cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), target_dim))
    else:
        image = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

    # Normalize image.
    if normalize:
        image = normalize_image(image)

    return image

def add_channel_dim(array: np.array) -> np.array:
    assert len(array.shape) == 2
    # Add the channel dimension to comply with Torch.
    array = array[None, :, :]
    return array

def get_time(path: str) -> float:
    ''' Get the timestamp information from a path string. '''
    time = os.path.basename(path).replace('week_', '').split('-')[0].replace('.png', '')
    # Shall be 2 or 3 digits
    assert len(time) in [2, 3]
    time = float(time)
    return time


class BrainLUMIEREDataset(Dataset):

    def __init__(self,
                 base_path: str = root_dir + '/data/brain_LUMIERE/',
                 image_folder: str = 'LUMIERE_images_tumor1200px_256x256/',
                 max_slice_per_patient: int = 20,
                 target_dim: Tuple[int] = (256, 256)):
        '''
        The special thing here is that different patients may have different number of visits.
        - If a patient has fewer than 2 visits, we ignore the patient.
        - When a patient's index is queried, we return images from all visits of that patient.
        - We need to be extra cautious that the data is split on the patient level rather than image pair level.

        NOTE: since different patients may have different number of visits, the returned array will
        not necessarily be of the same shape. Due to the concatenation requirements, we can only
        set batch size to 1 in the downstream Dataloader.

        NOTE: This dataset is structured like this:
        LUMIERE_images_final_256x256
        -- Patient-XX
            -- slice_YY
                -- week_ZZ.png

        LUMIERE_masks_final_256x256
        -- Patient-XX
            -- slice_YY
                -- week_ZZ_LUMIERE_mask.png

        So we will organize outputs on the unit of slices.
        Each slice is essentially treated as a separate trajectory.
        But importantly, data partitioning is done on the unit of patients.
        '''
        super().__init__()

        self.target_dim = target_dim
        self.max_slice_per_patient = max_slice_per_patient

        self.all_patient_folders = sorted(glob('%s/%s/*/' % (base_path, image_folder)))
        self.all_patient_ids = [os.path.basename(item.rstrip('/')) for item in self.all_patient_folders]
        self.patient_id_to_slice_id = []  # maps the patient id to a list of corresponding slice ids.

        self.image_by_slice = []

        self.max_t = 0

        curr_slice_idx = 0
        for folder in self.all_patient_folders:

            num_slices_curr_patient = 0
            slice_arr = np.array(sorted(glob('%s/slice*/' % (folder))))

            if self.max_slice_per_patient is not None \
                and len(slice_arr) > self.max_slice_per_patient:
                subset_ids = np.linspace(0, len(slice_arr)-1, self.max_slice_per_patient)
                subset_ids = np.array([int(item) for item in subset_ids])
                slice_arr = slice_arr[subset_ids]

            for curr_slice in slice_arr:
                paths = sorted(glob('%s/week*.png' % curr_slice))

                '''
                Ignore week 0!!!
                Week 0 is pre-operation, which means tumors will be cut!
                This dynamics may be too complicated to learn.
                If we ignore week 0, the remaining will likely be natural growth of tumor.
                '''
                paths = [p for p in paths if 'week_000' not in p]

                if len(paths) >= 2:
                    self.image_by_slice.append(paths)
                    num_slices_curr_patient += 1
                for p in paths:
                    self.max_t = max(self.max_t, get_time(p))

            self.patient_id_to_slice_id.append(np.arange(curr_slice_idx, curr_slice_idx + num_slices_curr_patient))
            curr_slice_idx += num_slices_curr_patient


    def return_statistics(self) -> None:
        print('max time (weeks):', self.max_t)

        unique_patient_list = np.unique(self.all_patient_ids)
        print('Number of unique patients:', len(unique_patient_list))
        print('Number of unique slices:', len(self.image_by_slice))

        num_visit_map = {}
        for item in self.image_by_slice:
            num_visit = len(item)
            if num_visit not in num_visit_map.keys():
                num_visit_map[num_visit] = 1
            else:
                num_visit_map[num_visit] += 1
        for k, v in sorted(num_visit_map.items()):
            print('%d visits: %d slices.' % (k, v))
        return

    def __len__(self) -> int:
        return len(self.all_patient_ids)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 1


class BrainLUMIERESubset(BrainLUMIEREDataset):

    def __init__(self,
                 main_dataset: BrainLUMIEREDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal['one_pair', 'all_pairs', 'all_subsequences', 'all_subarrays', 'full_sequence'],
                 transforms = None,
                 transforms_aug = None):
        '''
        A subset of BrainLUMIEREDataset.

        In BrainLUMIEREDataset, we carefully isolated the (variable number of) images from
        different patients, and in train/val/test split we split the data by
        patient rather than by image.

        Now we have 3 instances of BrainLUMIERESubset, one for each train/val/test set.
        In each set, we can safely unpack the images out.
        We want to organize the images such that each time `__getitem__` is called,
        it gets a pair of [x_start, x_end] and [t_start, t_end].
        '''
        super().__init__()

        self.target_dim = main_dataset.target_dim
        self.return_format = return_format
        self.transforms = transforms
        self.transforms_aug = transforms_aug

        self.image_by_slice = []

        for patient_id in subset_indices:
            slice_ids = main_dataset.patient_id_to_slice_id[patient_id]
            self.image_by_slice.extend([main_dataset.image_by_slice[i] for i in slice_ids])

        self.all_image_pairs = []
        self.all_subsequences = []
        self.all_subarrays = []
        for image_list in self.image_by_slice:
            pair_indices = list(itertools.combinations(np.arange(len(image_list)), r=2))
            for (idx1, idx2) in pair_indices:
                self.all_image_pairs.append(
                    [image_list[idx1], image_list[idx2]])
                self.all_subarrays.append(image_list[idx1 : idx2+1])

            for num_items in range(2, len(image_list)+1):
                subsequence_indices_list = list(itertools.combinations(np.arange(len(image_list)), r=num_items))
                for subsequence_indices in subsequence_indices_list:
                    self.all_subsequences.append([image_list[idx] for idx in subsequence_indices])

    def __len__(self) -> int:
        if self.return_format == 'one_pair':
            # If we only return 1 pair of images per patient...
            return len(self.image_by_slice)
        elif self.return_format == 'all_pairs':
            # If we return all pairs of images per patient...
            return len(self.all_image_pairs)
        elif self.return_format == 'all_subsequences':
            # If we return all subsequences of images per patient...
            return len(self.all_subsequences)
        elif self.return_format == 'all_subarrays':
            # If we return all subarrays of images per patient...
            return len(self.all_subarrays)
        elif self.return_format == 'full_sequence':
            # If we return the full sequences of images per patient...
            return len(self.image_by_slice)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        if self.return_format == 'one_pair':
            image_list = self.image_by_slice[idx]

            # ]
            rng = np.random.default_rng(1)

            image_list = self.image_by_slice[idx]

            pair_indices = list(
                itertools.combinations(np.arange(len(image_list)), r=2)
            )

            k = rng.integers(len(pair_indices))
            i, j = pair_indices[k]

            sampled_pair = [image_list[i], image_list[j]]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in sampled_pair
            ])
            timestamps = np.array([get_time(img) for img in sampled_pair])

        elif self.return_format == 'all_pairs':
            queried_pair = self.all_image_pairs[idx]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_pair
            ])
            timestamps = np.array([get_time(img) for img in queried_pair])

        elif self.return_format == 'all_subsequences':
            queried_sequence = self.all_subsequences[idx]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_sequence
            ])
            timestamps = np.array([get_time(img) for img in queried_sequence])

        elif self.return_format == 'all_subarrays':
            queried_sequence = self.all_subarrays[idx]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_sequence
            ])
            timestamps = np.array([get_time(img) for img in queried_sequence])

        elif self.return_format == 'full_sequence':
            queried_sequence = self.image_by_slice[idx]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_sequence
            ])
            timestamps = np.array([get_time(img) for img in queried_sequence])

        if self.return_format in ['one_pair', 'all_pairs']:
            assert len(images) == 2
            image1, image2 = images[0], images[1]
            if self.transforms is not None:
                transformed = self.transforms(image=image1, image_other=image2)
                image1 = transformed["image"]
                image2 = transformed["image_other"]

            if self.transforms_aug is not None:
                transformed_aug = self.transforms_aug(image=image1, image_other=image1)
                image1_aug = transformed_aug["image"]
                image1_aug = normalize_image(image1_aug)
                image1_aug = add_channel_dim(image1_aug)

            image1 = normalize_image(image1)
            image2 = normalize_image(image2)

            image1 = add_channel_dim(image1)
            image2 = add_channel_dim(image2)

            if self.transforms_aug is not None:
                images = np.vstack((image1[None, ...], image2[None, ...], image1_aug[None, ...]))
            else:
                images = np.vstack((image1[None, ...], image2[None, ...]))

        elif self.return_format in ['all_subsequences', 'all_subarrays', 'full_sequence']:
            num_images = len(images)
            assert num_images >= 2
            assert num_images < 20  # NOTE: see `additional_targets` in `transform`.

            # Unpack the subsequence.
            image_list = np.rollaxis(images, axis=0)

            data_dict = {'image': image_list[0]}
            for idx in range(num_images - 1):
                data_dict['image_other%d' % (idx + 1)] = image_list[idx + 1]

            if self.transforms is not None:
                data_dict = self.transforms(**data_dict)

            images = normalize_image(add_channel_dim(data_dict['image']))[None, ...]
            for idx in range(num_images - 1):
                images = np.vstack((images,
                                    normalize_image(add_channel_dim(data_dict['image_other%d' % (idx + 1)]))[None, ...]))
        return images, timestamps


class BrainLUMIERESegDataset(Dataset):

    def __init__(self,
                 base_path: str = root_dir + '/data/brain_LUMIERE/',
                 image_folder: str = 'LUMIERE_images_tumor1200px_256x256/',
                 mask_folder: str = 'LUMIERE_masks_tumor1200px_256x256/',
                 max_slice_per_patient: int = 20,
                 target_dim: Tuple[int] = (256, 256)):
        '''
        This dataset is for segmentation.
        '''
        super().__init__()

        self.target_dim = target_dim
        self.max_slice_per_patient = max_slice_per_patient

        all_patient_folders = sorted(glob('%s/%s/Patient-*/' % (base_path, image_folder)))

        self.image_by_patient = []
        self.mask_by_patient = []

        for patient_folder in all_patient_folders:
            curr_patient_slice_folders = np.array(sorted(glob('%s/slice*/' % patient_folder)))

            if self.max_slice_per_patient is not None \
                and len(curr_patient_slice_folders) > self.max_slice_per_patient:
                subset_ids = np.linspace(0, len(curr_patient_slice_folders)-1, self.max_slice_per_patient)
                subset_ids = np.array([int(item) for item in subset_ids])
                curr_patient_slice_folders = curr_patient_slice_folders[subset_ids]

            for im_folder in curr_patient_slice_folders:
                image_paths = sorted(glob('%s/*.png' % im_folder))
                mask_paths = []
                for image_path_ in image_paths:
                    mask_path_ = image_path_.replace('.png', '').replace(
                        image_folder, mask_folder) + '_LUMIERE_mask.png'
                    assert os.path.isfile(mask_path_)
                    mask_paths.append(mask_path_)
                self.image_by_patient.append(image_paths)
                self.mask_by_patient.append(mask_paths)

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 1


class BrainLUMIERESegSubset(BrainLUMIERESegDataset):

    def __init__(self,
                 main_dataset: BrainLUMIERESegDataset = None,
                 subset_indices: List[int] = None,
                 transforms = None):
        '''
        A subset of BrainLUMIERESegDataset.
        '''
        super().__init__()

        self.target_dim = main_dataset.target_dim

        image_by_patient = [
            main_dataset.image_by_patient[i] for i in subset_indices
        ]
        mask_by_patient = [
            main_dataset.mask_by_patient[i] for i in subset_indices
        ]

        self.image_list = [image for patient_folder in image_by_patient for image in patient_folder]
        self.mask_list = [mask for patient_folder in mask_by_patient for mask in patient_folder]
        assert len(self.image_list) == len(self.mask_list)

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = load_image(self.image_list[idx], target_dim=self.target_dim, normalize=False)
        mask = load_image(self.mask_list[idx], target_dim=self.target_dim, normalize=False)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = normalize_image(image)

        # I believe this means necrosis and contrast enhancement.
        # necrosis: 85, contrast enhancement: 170, edema: 255.
        assert mask.min() == 0 and mask.max() <= 255
        whole_msk = True #False #
        if whole_msk:
            mask = mask > 0
        else:
            mask = np.logical_and(mask > 0, mask < 250)

        image = add_channel_dim(image)
        mask = add_channel_dim(mask)

        return image, mask

def signed_distance(mask: np.ndarray):
    mask = mask.astype(bool)
    return distance_transform_edt(~mask) - distance_transform_edt(mask)


def compute_vx_vy_from_masks_numpy(
    m0: np.ndarray,
    m1: np.ndarray,
    delta_t: float,
    sd_flow_tau: float = 6.0,
    flow_scale: float = 3.0,
    vel_sigma: float = 1.0,
    sd_band_width: float = 10.0,
):
    """
    m0, m1: [H,W] binary mask (0/1)
    returns vx, vy: [H,W]
    """

    # signed distance
    sd0 = signed_distance(m0 > 0)
    sd1 = signed_distance(m1 > 0)

    # geometry proxy
    g0 = np.tanh(-sd0 / sd_flow_tau).astype(np.float32)
    g1 = np.tanh(-sd1 / sd_flow_tau).astype(np.float32)

    # optical flow (backward: g1 -> g0)
    v_back, u_back = optical_flow_tvl1(g1, g0)

    vy = -v_back / delta_t * flow_scale
    vx = -u_back / delta_t * flow_scale

    # smooth velocity
    vx = gaussian_filter(vx, vel_sigma)
    vy = gaussian_filter(vy, vel_sigma)

    # band support
    band = np.abs(sd0) < sd_band_width
    vx *= band
    vy *= band

    return vx.astype(np.float32), vy.astype(np.float32)

class BrainLUMIEREMaskDataset_ori(Dataset):
    """
    LUMIERE longitudinal dataset with IMAGE + MASK trajectories.
    Slice-level trajectories, patient-level split.
    """

    def __init__(self,
                 base_path: str = root_dir + '/data/brain_LUMIERE/',
                 image_folder: str = 'LUMIERE_images_tumor1200px_256x256/',
                 mask_folder: str  = 'LUMIERE_masks_tumor1200px_256x256/',
                 max_slice_per_patient: int = 20,
                 target_dim: Tuple[int] = (256, 256),
                 tau: float = 2.0,
                 sd_flow_tau: float = 6.0,
                 flow_scale: float = 3.0,
                 vel_sigma: float = 1.0,
                 sd_band_width: float = 10.0):

        super().__init__()

        self.target_dim = target_dim
        self.max_slice_per_patient = max_slice_per_patient

        self.all_patient_folders = sorted(
            glob(f'{base_path}/{image_folder}/Patient-*/')
        )
        self.all_patient_ids = [
            os.path.basename(p.rstrip('/'))
            for p in self.all_patient_folders
        ]

        self.patient_id_to_slice_id = []
        self.image_by_slice = []
        self.mask_by_slice  = []
        self.max_t = 0.0

        curr_slice_idx = 0

        # --------------------------------------------------
        # build slice-level trajectories
        # --------------------------------------------------
        for patient_folder in self.all_patient_folders:

            slice_folders = np.array(
                sorted(glob(f'{patient_folder}/slice*/'))
            )

            if self.max_slice_per_patient is not None and \
               len(slice_folders) > self.max_slice_per_patient:
                subset_ids = np.linspace(
                    0, len(slice_folders) - 1, self.max_slice_per_patient
                ).astype(int)
                slice_folders = slice_folders[subset_ids]

            num_slices_curr_patient = 0

            for slice_folder in slice_folders:
                img_paths = sorted(glob(f'{slice_folder}/week_*.png'))
                img_paths = [p for p in img_paths if 'week_000' not in p]

                msk_paths = [
                    p.replace(image_folder, mask_folder)
                     .replace('.png', '_LUMIERE_mask.png')
                    for p in img_paths
                ]

                if len(img_paths) >= 2:
                    for mp in msk_paths:
                        assert os.path.isfile(mp), f"Missing mask: {mp}"

                    self.image_by_slice.append(img_paths)
                    self.mask_by_slice.append(msk_paths)
                    num_slices_curr_patient += 1

                for p in img_paths:
                    self.max_t = max(self.max_t, get_time(p))

            self.patient_id_to_slice_id.append(
                np.arange(curr_slice_idx,
                          curr_slice_idx + num_slices_curr_patient)
            )
            curr_slice_idx += num_slices_curr_patient

    def __len__(self):
        return len(self.all_patient_ids)

    def num_image_channel(self) -> int:
        return 1
    

class BrainLUMIEREMaskSubset_ori(BrainLUMIEREMaskDataset_ori):
    """
    Subset for PDF on LUMIERE / brain_LUMIERE.
    Returns (images, masks, timestamps).
    """

    def __init__(self,
                #  main_dataset: BrainLUMIEREMaskDataset,
                #  subset_indices: List[int],
                 main_dataset: BrainLUMIEREMaskDataset_ori = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal[
                     'one_pair',
                     'all_pairs',
                     'all_subsequences',
                     'all_subarrays',
                     'full_sequence'
                 ],
                 transforms=None,
                 transforms_aug=None):

        super().__init__()

        self.target_dim = main_dataset.target_dim
        self.transforms = transforms

        self.image_by_slice = []
        self.mask_by_slice  = []

        for pid in subset_indices:
            slice_ids = main_dataset.patient_id_to_slice_id[pid]
            self.image_by_slice.extend(
                [main_dataset.image_by_slice[i] for i in slice_ids]
            )
            self.mask_by_slice.extend(
                [main_dataset.mask_by_slice[i] for i in slice_ids]
            )

    def __len__(self):
        return len(self.image_by_slice)

    def __getitem__(self, idx):
        """
        Returns:
            images: [2, 1, H, W]
            masks:  [2, 1, H, W]
            times:  [2]   (weeks)
        """

        img_paths = self.image_by_slice[idx]
        msk_paths = self.mask_by_slice[idx]

        # sample random pair (same as BrainLUMIERESubset one_pair)
        pair_idx = np.random.choice(len(img_paths), size=2, replace=False)
        pair_idx = np.sort(pair_idx)

        img_paths = [img_paths[i] for i in pair_idx]
        msk_paths = [msk_paths[i] for i in pair_idx]

        images = []
        masks  = []
        times  = []

        for ip, mp in zip(img_paths, msk_paths):
            img = load_image(ip, target_dim=self.target_dim, normalize=False)
            img = normalize_image(img)
            img = add_channel_dim(img)

            msk = load_image(mp, target_dim=self.target_dim, normalize=False)
            # LUMIERE lesion core: necrosis + enhancement
            # (consistent with BrainLUMIERESegDataset)
            msk = np.logical_and(msk > 0, msk < 250).astype(np.float32)
            msk = add_channel_dim(msk)

            images.append(img)
            masks.append(msk)
            times.append(get_time(ip))

        images = np.stack(images, axis=0)
        masks  = np.stack(masks, axis=0)
        times  = np.array(times, dtype=np.float32)

        return images, masks, times


############### flow version #################


class BrainLUMIEREMaskDataset_1(Dataset):
    """
    LUMIERE longitudinal dataset with IMAGE + MASK trajectories.
    Slice-level trajectories, patient-level split.
    """

    def __init__(self,
                 base_path: str = root_dir + '/data/brain_LUMIERE/',
                 image_folder: str = 'LUMIERE_images_tumor1200px_256x256/',
                 mask_folder: str  = 'LUMIERE_masks_tumor1200px_256x256/',
                 max_slice_per_patient: int = 20,
                 target_dim: Tuple[int] = (256, 256),
                 tau: float = 2.0,
                 sd_flow_tau: float = 6.0,
                 flow_scale: float = 3.0,
                 vel_sigma: float = 1.0,
                 sd_band_width: float = 10.0):

        super().__init__()

        self.target_dim = target_dim
        self.max_slice_per_patient = max_slice_per_patient

        self.all_patient_folders = sorted(
            glob(f'{base_path}/{image_folder}/Patient-*/')
        )
        self.all_patient_ids = [
            os.path.basename(p.rstrip('/'))
            for p in self.all_patient_folders
        ]

        self.patient_id_to_slice_id = []
        self.image_by_slice = []
        self.mask_by_slice  = []
        self.time_by_slice  = []

        self.flow_by_slice = []   # (slice_idx, t0_idx, t1_idx) -> (vx, vy)

        self.max_t = 0.0

        self.flow_root = os.path.join(base_path, "flow_cache_whole")
        os.makedirs(self.flow_root, exist_ok=True)

        curr_slice_idx = 0

        # --------------------------------------------------
        # build slice-level trajectories
        # --------------------------------------------------
        for patient_folder in tqdm(self.all_patient_folders):

            slice_folders = np.array(
                sorted(glob(f'{patient_folder}/slice*/'))
            )

            if self.max_slice_per_patient is not None and \
               len(slice_folders) > self.max_slice_per_patient:
                subset_ids = np.linspace(
                    0, len(slice_folders) - 1, self.max_slice_per_patient
                ).astype(int)
                slice_folders = slice_folders[subset_ids]

            num_slices_curr_patient = 0

            for slice_folder in slice_folders:
                img_paths = sorted(glob(f'{slice_folder}/week_*.png'))
                img_paths = [p for p in img_paths if 'week_000' not in p]

                msk_paths = [
                    p.replace(image_folder, mask_folder)
                     .replace('.png', '_LUMIERE_mask.png')
                    for p in img_paths
                ]

                if len(img_paths) >= 2:
                    for mp in msk_paths:
                        assert os.path.isfile(mp), f"Missing mask: {mp}"

                    self.image_by_slice.append(img_paths)
                    self.mask_by_slice.append(msk_paths)
                    num_slices_curr_patient += 1

                    times = [get_time(p) for p in img_paths]
                    self.time_by_slice.append(times)

                    # --------------------------------------------------
                    # PRECOMPUTE FLOW FOR THIS SLICE
                    # --------------------------------------------------
                    slice_flow_dir = os.path.join(
                        self.flow_root,
                        slice_folder.split('/')[-3],
                        slice_folder.split('/')[-2]
                    )
                    os.makedirs(slice_flow_dir, exist_ok=True)

                    # load all masks ONCE
                    masks_np = []
                    for mp in msk_paths:
                        m = load_image(mp, target_dim=self.target_dim)
                        m = (m > 0).astype(np.uint8)
                        masks_np.append(m)

                    T = len(masks_np)

                    flow_paths = []
                    for i in range(T):
                        for j in range(i + 1, T):

                            flow_path = os.path.join(
                                slice_flow_dir,
                                f"flow_t{i}_t{j}.npz"
                            )

                            flow_paths.append(flow_path)

                            if os.path.isfile(flow_path):
                                continue

                            delta_t = max(times[j] - times[i], 1e-4)

                            # ---------- compute vx, vy (same as your analysis) ----------
                            sd0 = signed_distance(masks_np[i] > 0)
                            sd1 = signed_distance(masks_np[j] > 0)
                            # only consider two categories
                            

                            g0 = np.tanh(-sd0 / sd_flow_tau).astype(np.float32)
                            g1 = np.tanh(-sd1 / sd_flow_tau).astype(np.float32)

                            v_back, u_back = optical_flow_tvl1(g1, g0)

                            vy = -v_back / delta_t * flow_scale
                            vx = -u_back / delta_t * flow_scale

                            vx = gaussian_filter(vx, vel_sigma)
                            vy = gaussian_filter(vy, vel_sigma)

                            band = np.abs(sd0) < sd_band_width
                            vx *= band
                            vy *= band

                            # ---------- save ----------
                            np.savez_compressed(
                                flow_path,
                                vx=vx.astype(np.float32),
                                vy=vy.astype(np.float32),
                                t0=times[i],
                                t1=times[j],
                            )
                    
                    self.flow_by_slice.append(flow_path)


                for p in img_paths:
                    self.max_t = max(self.max_t, get_time(p))

            self.patient_id_to_slice_id.append(
                np.arange(curr_slice_idx,
                          curr_slice_idx + num_slices_curr_patient)
            )
            curr_slice_idx += num_slices_curr_patient

    def __len__(self):
        return len(self.all_patient_ids)

    def num_image_channel(self) -> int:
        return 1


class BrainLUMIEREMaskDataset(Dataset):
    """
    LUMIERE longitudinal dataset with IMAGE + MASK trajectories.
    Slice-level trajectories, patient-level split.
    """

    def __init__(self,
                base_path: str = root_dir + '/data/brain_LUMIERE/',
                image_folder: str = 'LUMIERE_images_tumor1200px_256x256/',
                mask_folder: str  = 'LUMIERE_masks_tumor1200px_256x256/',
                max_slice_per_patient: int = 20,
                target_dim: Tuple[int] = (256, 256),
                tau: float = 2.0,
                sd_flow_tau: float = 6.0,   # kept for compat, not used
                flow_scale: float = 3.0,
                vel_sigma: float = 1.0,
                sd_band_width: float = 10.0):  # kept for compat, not used

        super().__init__()

        self.target_dim = target_dim
        self.max_slice_per_patient = max_slice_per_patient

        self.all_patient_folders = sorted(
            glob(f'{base_path}/{image_folder}/Patient-*/')
        )
        self.all_patient_ids = [
            os.path.basename(p.rstrip('/'))
            for p in self.all_patient_folders
        ]

        self.patient_id_to_slice_id = []
        self.image_by_slice = []
        self.mask_by_slice  = []
        self.time_by_slice  = []

        self.flow_by_slice = []

        self.max_t = 0.0

        self.flow_root = os.path.join(base_path, "flow_cache_2")
        os.makedirs(self.flow_root, exist_ok=True)

        curr_slice_idx = 0

        # --------------------------------------------------
        # New flow hyperparameters.
        # --------------------------------------------------
        sd_clip    = 12.0
        soft_width = 18.0
        soft_sharp = 6.0

        # --------------------------------------------------
        # --------------------------------------------------
        def sd_proxy(sd):
            """
            clipped linear SD proxy in [0,1], tumor brighter
            """
            s = np.clip(sd, -sd_clip, sd_clip)
            s = (-s - s.min()) / (s.max() - s.min() + 1e-8)
            return s.astype(np.float32)

        def soft_support(sd):
            """
            continuous spatial support (no donut)
            """
            w = 1.0 / (1.0 + np.exp((sd - soft_width) / soft_sharp))
            return gaussian_filter(w, 1.0)

        # --------------------------------------------------
        # build slice-level trajectories
        # --------------------------------------------------
        for patient_folder in tqdm(self.all_patient_folders):

            slice_folders = np.array(
                sorted(glob(f'{patient_folder}/slice*/'))
            )

            if self.max_slice_per_patient is not None and \
            len(slice_folders) > self.max_slice_per_patient:
                subset_ids = np.linspace(
                    0, len(slice_folders) - 1, self.max_slice_per_patient
                ).astype(int)
                slice_folders = slice_folders[subset_ids]

            num_slices_curr_patient = 0

            for slice_folder in slice_folders:

                img_paths = sorted(glob(f'{slice_folder}/week_*.png'))
                img_paths = [p for p in img_paths if 'week_000' not in p]

                msk_paths = [
                    p.replace(image_folder, mask_folder)
                    .replace('.png', '_LUMIERE_mask.png')
                    for p in img_paths
                ]

                if len(img_paths) >= 2:
                    for mp in msk_paths:
                        assert os.path.isfile(mp), f"Missing mask: {mp}"

                    self.image_by_slice.append(img_paths)
                    self.mask_by_slice.append(msk_paths)
                    num_slices_curr_patient += 1

                    times = [get_time(p) for p in img_paths]
                    self.time_by_slice.append(times)

                    # --------------------------------------------------
                    # PRECOMPUTE FLOW FOR THIS SLICE
                    # --------------------------------------------------
                    slice_flow_dir = os.path.join(
                        self.flow_root,
                        slice_folder.split('/')[-3],
                        slice_folder.split('/')[-2]
                    )
                    os.makedirs(slice_flow_dir, exist_ok=True)

                    # load all masks ONCE
                    masks_np = []
                    for mp in msk_paths:
                        m = load_image(mp, target_dim=self.target_dim)
                        masks_np.append((m > 0).astype(np.uint8))

                    T = len(masks_np)

                    for i in range(T):
                        for j in range(i + 1, T):

                            flow_path = os.path.join(
                                slice_flow_dir,
                                f"flow_t{i}_t{j}.npz"
                            )

                            if os.path.isfile(flow_path):
                                continue

                            delta_t = max(times[j] - times[i], 1e-4)

                            # --------------------------------------------------
                            # New flow.
                            # --------------------------------------------------
                            sd0 = signed_distance(masks_np[i])
                            sd1 = signed_distance(masks_np[j])

                            g0 = sd_proxy(sd0)
                            g1 = sd_proxy(sd1)

                            v_back, u_back = optical_flow_tvl1(g1, g0)

                            vy = -v_back / delta_t * flow_scale
                            vx = -u_back / delta_t * flow_scale

                            vx = gaussian_filter(vx, vel_sigma)
                            vy = gaussian_filter(vy, vel_sigma)

                            # SOFT support (no donut)
                            w = soft_support(sd0)
                            vx *= w
                            vy *= w

                            # --------------------------------------------------
                            # save (UNCHANGED FORMAT)
                            # --------------------------------------------------
                            np.savez_compressed(
                                flow_path,
                                vx=vx.astype(np.float32),
                                vy=vy.astype(np.float32),
                                t0=times[i],
                                t1=times[j],
                            )

                    self.flow_by_slice.append(slice_flow_dir)

                for p in img_paths:
                    self.max_t = max(self.max_t, get_time(p))

            self.patient_id_to_slice_id.append(
                np.arange(curr_slice_idx,
                        curr_slice_idx + num_slices_curr_patient)
            )
            curr_slice_idx += num_slices_curr_patient


    def __len__(self):
        return len(self.all_patient_ids)

    def num_image_channel(self) -> int:
        return 1

class BrainLUMIEREMaskSubset(BrainLUMIEREMaskDataset):
    """
    Subset for PDF on LUMIERE / brain_LUMIERE.
    Returns (images, masks, timestamps).
    """

    def __init__(self,
                #  main_dataset: BrainLUMIEREMaskDataset,
                #  subset_indices: List[int],
                 main_dataset: BrainLUMIEREMaskDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal[
                     'one_pair',
                     'all_pairs',
                     'all_subsequences',
                     'all_subarrays',
                     'full_sequence'
                 ],
                 transforms=None,
                 transforms_aug=None):

        super().__init__()

        self.target_dim = main_dataset.target_dim
        self.transforms = transforms

        self.image_by_slice = []
        self.mask_by_slice  = []
        self.time_by_slice  = []

        for pid in subset_indices:
            slice_ids = main_dataset.patient_id_to_slice_id[pid]
            self.image_by_slice.extend(
                [main_dataset.image_by_slice[i] for i in slice_ids]
            )
            self.mask_by_slice.extend(
                [main_dataset.mask_by_slice[i] for i in slice_ids]
            )
            self.time_by_slice.extend(
                [main_dataset.time_by_slice[i] for i in slice_ids]
            )

    def __len__(self):
        return len(self.image_by_slice)

    def __getitem__(self, idx):
        """
        Returns:
            images: [2, 1, H, W]
            masks:  [2, 1, H, W]
            times:  [2]   (weeks)
        """

        img_paths = self.image_by_slice[idx]
        msk_paths = self.mask_by_slice[idx]
        times_all = self.time_by_slice[idx]

        # sample random pair (same as BrainLUMIERESubset one_pair)


        rng = np.random.default_rng(1)

        pair_indices = list(itertools.combinations(np.arange(len(img_paths)), r=2))
        k = rng.integers(len(pair_indices))
        i, j = pair_indices[k]


        images = []
        masks  = []

        for ip, mp in zip([img_paths[i], img_paths[j]],
                          [msk_paths[i], msk_paths[j]]):

            img = load_image(ip, target_dim=self.target_dim, normalize=False)
            img = normalize_image(img)
            img = add_channel_dim(img)

            msk = load_image(mp, target_dim=self.target_dim, normalize=False)
            # LUMIERE lesion core: necrosis + enhancement
            # (consistent with BrainLUMIERESegDataset)
            whole_msk = True #False #
            if whole_msk:
                msk = msk > 0
            else:
                msk = np.logical_and(msk > 0, msk < 250).astype(np.float32)
            msk = add_channel_dim(msk)

            images.append(img)
            masks.append(msk)

        images = np.stack(images, axis=0)
        masks  = np.stack(masks, axis=0)
        times  = np.array([times_all[i], times_all[j]], dtype=np.float32)

        # --------------------------------------------------
        # fetch precomputed vx / vy (CRITICAL PART)
        # --------------------------------------------------
        flow_path = os.path.join(
                    self.flow_root,
                    img_paths[i].split('/')[-3],
                    img_paths[i].split('/')[-2],
                    f"flow_t{i}_t{j}.npz"
                )
        flow_data = np.load(flow_path)
        vx = flow_data["vx"][None, None, ...]   # [1,1,H,W]
        vy = flow_data["vy"][None, None, ...]


        return images, masks, times, vx, vy







if __name__ == '__main__':
    print('Full set.')
    dataset = BrainLUMIEREDataset(max_slice_per_patient=None)
    dataset.return_statistics()

    print('Subset with max 20 slices per patient.')
    dataset = BrainLUMIEREDataset(max_slice_per_patient=20)
    dataset.return_statistics()
