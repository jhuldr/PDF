"""
A longitudinal UCSF brain dataset.

This loader adapts the longitudinal dataset interface to the UCSF data format and clinical time definition.
"""

import itertools
from typing import List, Tuple, Literal
from glob import glob
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from scipy.optimize import nnls
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.registration import optical_flow_tvl1

root_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])


# ======================================================
# Image utils (same as LUMIERE)
# ======================================================
def normalize_image(image: np.array) -> np.array:
    """
    Image already normalized on scan level.
    Transform to [-1, 1].
    """
    assert image.min() >= 0 and image.max() <= 255
    image = image / 255.0 * 2 - 1
    image = np.clip(image, -1.0, 1.0)
    return image


def load_image(path: str,
               target_dim: Tuple[int] = None,
               normalize: bool = True) -> np.array:
    """Load image as numpy array."""
    if target_dim is not None:
        image = np.array(
            cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), target_dim))
    else:
        image = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

    if normalize:
        image = normalize_image(image)

    return image


def add_channel_dim(array: np.array) -> np.array:
    assert len(array.shape) == 2
    return array[None, :, :]


# ======================================================
# Time parsing (UCSF-specific)
# ======================================================
def get_time(path: str) -> float:
    """
    Parse time (days) from UCSF file name.

    Example:
        time1_101.png -> 101
        time2_145.png -> 145
    """
    base = os.path.basename(path)
    # remove extension
    base = base.replace('.png', '')
    # split: time2_145
    parts = base.split('_')
    assert parts[0].startswith('time')
    time_days = float(parts[1])
    return int(round(time_days / 7.0))


# ======================================================
# ======================================================
class BrainUCSFDataset(Dataset):

    def __init__(self,
                 base_path: str = root_dir + '/data/brain_UCSF/',
                 image_folder: str = 'brain_UCSF_images_axial_tumor500px_256x256/',
                 max_slice_per_patient: int = 20,
                 target_dim: Tuple[int] = (256, 256)):
        """
        UCSF longitudinal dataset.

        - Each slice is treated as an independent temporal trajectory.
        - Different patients may have different number of visits.
        - Data split MUST be done on patient level.

        Folder structure:
        brain_UCSF_images_axial_...
        └── PatientID
            └── slice_XX
                ├── time1_101.png
                ├── time2_145.png
                └── ...
        """

        super().__init__()

        self.target_dim = target_dim
        self.max_slice_per_patient = max_slice_per_patient

        self.all_patient_folders = sorted(
            glob(f'{base_path}/{image_folder}/*/')
        )
        self.all_patient_ids = [
            os.path.basename(item.rstrip('/'))
            for item in self.all_patient_folders
        ]

        self.patient_id_to_slice_id = []
        self.image_by_slice = []
        self.max_t = 0.0

        curr_slice_idx = 0

        # --------------------------------------------------
        # Build slice-level trajectories
        # --------------------------------------------------
        for folder in self.all_patient_folders:

            num_slices_curr_patient = 0
            slice_arr = np.array(sorted(glob(f'{folder}/slice*/')))

            # Optional subsampling of slices per patient
            if self.max_slice_per_patient is not None and \
               len(slice_arr) > self.max_slice_per_patient:
                subset_ids = np.linspace(
                    0, len(slice_arr) - 1, self.max_slice_per_patient
                ).astype(int)
                slice_arr = slice_arr[subset_ids]

            for curr_slice in slice_arr:
                # UCSF uses timeX_YYY.png
                paths = sorted(glob(f'{curr_slice}/time*.png'))

                # Require at least two time points
                if len(paths) >= 2:
                    self.image_by_slice.append(paths)
                    num_slices_curr_patient += 1

                # Track maximum time (days)
                for p in paths:
                    self.max_t = max(self.max_t, get_time(p))

            self.patient_id_to_slice_id.append(
                np.arange(curr_slice_idx,
                          curr_slice_idx + num_slices_curr_patient)
            )
            curr_slice_idx += num_slices_curr_patient

    # ==================================================
    # ==================================================
    def return_statistics(self) -> None:
        print('max time (days):', self.max_t)
        print('Number of unique patients:', len(self.all_patient_ids))
        print('Number of unique slices:', len(self.image_by_slice))

        num_visit_map = {}
        for item in self.image_by_slice:
            num_visit = len(item)
            num_visit_map[num_visit] = num_visit_map.get(num_visit, 0) + 1

        for k, v in sorted(num_visit_map.items()):
            print(f'{k} visits: {v} slices.')

    def __len__(self) -> int:
        return len(self.all_patient_ids)

    def num_image_channel(self) -> int:
        return 1



class BrainUCSFSubset(BrainUCSFDataset):

    def __init__(self,
                 main_dataset: BrainUCSFDataset = None,
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
        """
        UCSF subset of BrainUCSFDataset.

        This class follows the common longitudinal subset interface, but:
        - time is measured in DAYS (real clinical time)
        - image paths are UCSF-style (timeX_YYY.png)

        All other semantics (slice-level trajectories, patient-level split,
        return formats) are identical to LUMIERE.
        """
        super().__init__()

        self.target_dim = main_dataset.target_dim
        self.return_format = return_format
        self.transforms = transforms
        self.transforms_aug = transforms_aug

        # Collect slice-level trajectories for this subset
        self.image_by_slice = []

        for patient_id in subset_indices:
            slice_ids = main_dataset.patient_id_to_slice_id[patient_id]
            self.image_by_slice.extend(
                [main_dataset.image_by_slice[i] for i in slice_ids]
            )

        # Precompute different sequence organizations
        self.all_image_pairs = []
        self.all_subsequences = []
        self.all_subarrays = []

        for image_list in self.image_by_slice:
            # all pairs
            pair_indices = list(
                itertools.combinations(np.arange(len(image_list)), r=2)
            )
            for idx1, idx2 in pair_indices:
                self.all_image_pairs.append(
                    [image_list[idx1], image_list[idx2]]
                )
                self.all_subarrays.append(image_list[idx1:idx2+1])

            # all subsequences
            for num_items in range(2, len(image_list) + 1):
                subseq_indices = list(
                    itertools.combinations(np.arange(len(image_list)), r=num_items)
                )
                for inds in subseq_indices:
                    self.all_subsequences.append(
                        [image_list[i] for i in inds]
                    )

    # --------------------------------------------------
    # --------------------------------------------------
    def __len__(self) -> int:
        if self.return_format == 'one_pair':
            return len(self.image_by_slice)
        elif self.return_format == 'all_pairs':
            return len(self.all_image_pairs)
        elif self.return_format == 'all_subsequences':
            return len(self.all_subsequences)
        elif self.return_format == 'all_subarrays':
            return len(self.all_subarrays)
        elif self.return_format == 'full_sequence':
            return len(self.image_by_slice)

    # --------------------------------------------------
    # Get item
    # --------------------------------------------------
    def __getitem__(self, idx) -> Tuple[np.array, np.array]:

        # --------------------------------------------------
        # --------------------------------------------------
        if self.return_format == 'one_pair':
            image_list = self.image_by_slice[idx]
            pair_indices = list(
                itertools.combinations(np.arange(len(image_list)), r=2)
            )

            sampled_pair = [
                image_list[i]
                for i in pair_indices[np.random.choice(len(pair_indices))]
            ]

            images = np.array([
                load_image(p, target_dim=self.target_dim, normalize=False)
                for p in sampled_pair
            ])
            timestamps = np.array([get_time(p) for p in sampled_pair], dtype=np.float32)

        elif self.return_format == 'all_pairs':
            queried_pair = self.all_image_pairs[idx]
            images = np.array([
                load_image(p, target_dim=self.target_dim, normalize=False)
                for p in queried_pair
            ])
            timestamps = np.array([get_time(p) for p in queried_pair], dtype=np.float32)

        elif self.return_format == 'all_subsequences':
            queried_sequence = self.all_subsequences[idx]
            images = np.array([
                load_image(p, target_dim=self.target_dim, normalize=False)
                for p in queried_sequence
            ])
            timestamps = np.array([get_time(p) for p in queried_sequence], dtype=np.float32)

        elif self.return_format == 'all_subarrays':
            queried_sequence = self.all_subarrays[idx]
            images = np.array([
                load_image(p, target_dim=self.target_dim, normalize=False)
                for p in queried_sequence
            ])
            timestamps = np.array([get_time(p) for p in queried_sequence], dtype=np.float32)

        elif self.return_format == 'full_sequence':
            queried_sequence = self.image_by_slice[idx]
            images = np.array([
                load_image(p, target_dim=self.target_dim, normalize=False)
                for p in queried_sequence
            ])
            timestamps = np.array([get_time(p) for p in queried_sequence], dtype=np.float32)

        # --------------------------------------------------
        # Pair-based formats (exactly same as LUMIERE)
        # --------------------------------------------------
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

            image1 = add_channel_dim(normalize_image(image1))
            image2 = add_channel_dim(normalize_image(image2))

            if self.transforms_aug is not None:
                images = np.vstack((
                    image1[None, ...],
                    image2[None, ...],
                    image1_aug[None, ...]
                ))
            else:
                images = np.vstack((
                    image1[None, ...],
                    image2[None, ...]
                ))

        # --------------------------------------------------
        # Sequence-based formats (exactly same as LUMIERE)
        # --------------------------------------------------
        elif self.return_format in ['all_subsequences', 'all_subarrays', 'full_sequence']:
            num_images = len(images)
            assert num_images >= 2
            assert num_images < 20

            image_list = np.rollaxis(images, axis=0)

            data_dict = {'image': image_list[0]}
            for i in range(num_images - 1):
                data_dict[f'image_other{i+1}'] = image_list[i+1]

            if self.transforms is not None:
                data_dict = self.transforms(**data_dict)

            images = normalize_image(add_channel_dim(data_dict['image']))[None, ...]
            for i in range(num_images - 1):
                images = np.vstack((
                    images,
                    normalize_image(add_channel_dim(data_dict[f'image_other{i+1}']))[None, ...]
                ))

        return images, timestamps



# ======================================================
# UCSF Segmentation Dataset
# ======================================================
class BrainUCSFSegDataset(Dataset):

    def __init__(self,
                 base_path: str = root_dir + '/data/brain_UCSF/',
                 image_folder: str = 'brain_UCSF_images_axial_tumor500px_256x256/',
                 mask_folder: str = 'brain_UCSF_masks_axial_tumor500px_256x256/',
                 max_slice_per_patient: int = 20,
                 target_dim: Tuple[int] = (256, 256)):
        """
        UCSF segmentation dataset.

        Each sample is a single image-mask pair.
        Slices are selected upstream; this dataset does NOT use time.
        """

        super().__init__()

        self.target_dim = target_dim
        self.max_slice_per_patient = max_slice_per_patient

        all_patient_folders = sorted(
            glob(f'{base_path}/{image_folder}/*/')
        )

        self.image_by_patient = []
        self.mask_by_patient = []

        for patient_folder in all_patient_folders:
            slice_folders = np.array(
                sorted(glob(f'{patient_folder}/slice*/'))
            )

            # Optional slice subsampling
            if self.max_slice_per_patient is not None and \
               len(slice_folders) > self.max_slice_per_patient:
                subset_ids = np.linspace(
                    0, len(slice_folders) - 1, self.max_slice_per_patient
                ).astype(int)
                slice_folders = slice_folders[subset_ids]

            for slice_folder in slice_folders:
                image_paths = sorted(glob(f'{slice_folder}/*.png'))

                mask_paths = []
                for img_p in image_paths:
                    mask_p = img_p.replace(
                        image_folder, mask_folder
                    ).replace('.png', '_mask.png')

                    assert os.path.isfile(mask_p), f'Missing mask: {mask_p}'
                    mask_paths.append(mask_p)

                self.image_by_patient.append(image_paths)
                self.mask_by_patient.append(mask_paths)

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        return 1


class BrainUCSFSegSubset(BrainUCSFSegDataset):

    def __init__(self,
                 main_dataset: BrainUCSFSegDataset = None,
                 subset_indices: List[int] = None,
                 transforms=None):
        """
        Subset of UCSF segmentation dataset.
        """

        super().__init__()

        self.target_dim = main_dataset.target_dim

        image_by_patient = [
            main_dataset.image_by_patient[i]
            for i in subset_indices
        ]
        mask_by_patient = [
            main_dataset.mask_by_patient[i]
            for i in subset_indices
        ]

        self.image_list = [
            img for patient in image_by_patient for img in patient
        ]
        self.mask_list = [
            msk for patient in mask_by_patient for msk in patient
        ]

        assert len(self.image_list) == len(self.mask_list)

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:

        image = load_image(
            self.image_list[idx],
            target_dim=self.target_dim,
            normalize=False
        )

        mask = load_image(
            self.mask_list[idx],
            target_dim=self.target_dim,
            normalize=False
        )

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Normalize image to [-1, 1]
        image = normalize_image(image)

        # --------------------------------------------------
        # UCSF tumor-core segmentation target
        # NCR (1) & ET (3) are positive; others negative
        # --------------------------------------------------
        # PNG values:
        # So keep values corresponding to label 1 and 3
        whole_msk = True #False #
        if whole_msk:
            mask = mask>0
        else:
            mask = np.logical_or(
                mask == (255 // 4) * 1,
                mask == (255 // 4) * 3,
            ).astype(np.float32)

        image = add_channel_dim(image)
        mask = add_channel_dim(mask.astype(np.float32))

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

class BrainUCSFMaskDataset_1(Dataset):
    """
    UCSF longitudinal dataset with IMAGE + MASK trajectories.
    Slice-level trajectories, patient-level split.
    """

    def __init__(self,
                 base_path: str = root_dir + '/data/brain_UCSF/',
                 image_folder: str = 'brain_UCSF_images_axial_tumor500px_256x256/',
                 mask_folder: str  = 'brain_UCSF_masks_axial_tumor500px_256x256/',
                 max_slice_per_patient: int = 20,
                 target_dim: Tuple[int] = (256, 256),
                 tau: float = 2.0,
                 sd_flow_tau: float = 6.0,
                 flow_scale: float = 3.0,
                 vel_sigma: float = 1.0,
                 sd_band_width: float = 10.0):

        super().__init__()

        self.target_dim = target_dim

        self.image_by_slice = []
        self.mask_by_slice  = []
        self.time_by_slice  = []

        self.flow_by_slice = []   # (slice_idx, t0_idx, t1_idx) -> (vx, vy)

        # --------------------------------------------------
        # discover patients / slices
        # --------------------------------------------------
        self.all_patient_folders = sorted(glob(f"{base_path}/{image_folder}/*/"))
        self.all_patient_ids = [ os.path.basename(p.rstrip('/')) for p in self.all_patient_folders ]

        self.patient_id_to_slice_id = [] 
        self.image_by_slice = [] # list[list[path]] 
        self.mask_by_slice = [] # list[list[path]] 

        self.flow_root = os.path.join(base_path, "flow_cache")
        os.makedirs(self.flow_root, exist_ok=True)

        self.max_t = 0.0

        curr_slice_idx = 0

        for pid, folder in tqdm(enumerate(self.all_patient_folders)):

            slice_folders = sorted(glob(f"{folder}/slice*/"))
            if max_slice_per_patient is not None:
                slice_folders = slice_folders[:max_slice_per_patient]

            num_slices_curr_patient = 0

            for slice_folder in slice_folders:

                img_paths = sorted(glob(f"{slice_folder}/time*.png"))
                img_paths = [p for p in img_paths if not p.endswith("_mask.png")]

                if len(img_paths) < 2:
                    continue

                msk_paths = [
                    p.replace(image_folder, mask_folder)
                    .replace(".png", "_mask.png")
                    for p in img_paths
                ]

                times = [get_time(p) for p in img_paths]

                # --------------------------------------------------
                # Register slice-level metadata.
                # --------------------------------------------------
                self.image_by_slice.append(img_paths)
                self.mask_by_slice.append(msk_paths)
                self.time_by_slice.append(times)

                for p in img_paths:
                    self.max_t = max(self.max_t, get_time(p))
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

                num_slices_curr_patient += 1
                curr_slice_idx += 1

            # map patient -> slice indices
            self.patient_id_to_slice_id.append(
                np.arange(curr_slice_idx - num_slices_curr_patient, curr_slice_idx)
            )

    def __len__(self):
        return len(self.all_patient_ids)
    
    def num_image_channel(self) -> int:
        return 1

class BrainUCSFMaskDataset(Dataset):
    """
    UCSF longitudinal dataset with IMAGE + MASK trajectories.
    Slice-level trajectories, patient-level split.
    """

    def __init__(self,
             base_path: str = root_dir + '/data/brain_UCSF/',
             image_folder: str = 'brain_UCSF_images_axial_tumor500px_256x256/',
             mask_folder: str  = 'brain_UCSF_masks_axial_tumor500px_256x256/',
             max_slice_per_patient: int = 20,
             target_dim: Tuple[int] = (256, 256),
             tau: float = 2.0,
             flow_scale: float = 3.0,
             vel_sigma: float = 1.0):

        super().__init__()

        self.target_dim = target_dim

        self.image_by_slice = []
        self.mask_by_slice  = []
        self.time_by_slice  = []
        self.flow_by_slice  = []

        # --------------------------------------------------
        # discover patients / slices
        # --------------------------------------------------
        self.all_patient_folders = sorted(glob(f"{base_path}/{image_folder}/*/"))
        self.all_patient_ids = [
            os.path.basename(p.rstrip('/')) for p in self.all_patient_folders
        ]

        self.patient_id_to_slice_id = []

        self.flow_root = os.path.join(base_path, "flow_cache_2")
        os.makedirs(self.flow_root, exist_ok=True)

        self.max_t = 0.0
        curr_slice_idx = 0

        # --------------------------------------------------
        # B-class flow hyperparameters (fixed)
        # --------------------------------------------------
        sigma_sd    = 4.0
        nsteps_rd  = 120
        sd_clip    = 12.0
        soft_width = 18.0
        soft_sharp = 6.0

        # --------------------------------------------------
        # --------------------------------------------------
        def sd_from_mask(mask):
            return gaussian_filter(signed_distance(mask), sigma=sigma_sd)

        def c_from_sd(sd):
            return 1.0 / (1.0 + np.exp(sd / tau))

        def sd_from_c(c, eps=1e-6):
            c = np.clip(c, eps, 1 - eps)
            return -tau * np.log(c / (1 - c))

        def sd_proxy(sd):
            s = np.clip(sd, -sd_clip, sd_clip)
            s = (-s - s.min()) / (s.max() - s.min() + 1e-8)
            return s.astype(np.float32)

        def soft_support(sd):
            w = 1.0 / (1.0 + np.exp((sd - soft_width) / soft_sharp))
            return gaussian_filter(w, 1.0)
        
        def laplacian_neumann(u: np.ndarray) -> np.ndarray:
            """
            2D Laplacian with Neumann (zero-flux) boundary condition
            """
            up = np.pad(u, 1, mode="edge")
            return (
                up[:-2, 1:-1] +
                up[2:,  1:-1] +
                up[1:-1, :-2] +
                up[1:-1,  2:] -
                4.0 * up[1:-1, 1:-1]
            )

        # --------------------------------------------------
        # main traversal
        # --------------------------------------------------
        for pid, folder in tqdm(enumerate(self.all_patient_folders)):

            slice_folders = sorted(glob(f"{folder}/slice*/"))
            if max_slice_per_patient is not None:
                slice_folders = slice_folders[:max_slice_per_patient]

            num_slices_curr_patient = 0

            for slice_folder in slice_folders:

                img_paths = sorted(glob(f"{slice_folder}/time*.png"))
                img_paths = [p for p in img_paths if not p.endswith("_mask.png")]

                if len(img_paths) < 2:
                    continue

                msk_paths = [
                    p.replace(image_folder, mask_folder)
                    .replace(".png", "_mask.png")
                    for p in img_paths
                ]

                times = [get_time(p) for p in img_paths]

                self.image_by_slice.append(img_paths)
                self.mask_by_slice.append(msk_paths)
                self.time_by_slice.append(times)

                for p in img_paths:
                    self.max_t = max(self.max_t, get_time(p))

                # --------------------------------------------------
                # flow cache directory
                # --------------------------------------------------
                slice_flow_dir = os.path.join(
                    self.flow_root,
                    slice_folder.split('/')[-3],
                    slice_folder.split('/')[-2]
                )
                os.makedirs(slice_flow_dir, exist_ok=True)

                # --------------------------------------------------
                # load masks
                # --------------------------------------------------
                masks_np = []
                for mp in msk_paths:
                    m = load_image(mp, target_dim=self.target_dim)
                    masks_np.append((m > 0).astype(np.uint8))

                T = len(masks_np)

                # --------------------------------------------------
                # pairwise B-class flow
                # --------------------------------------------------
                for i in range(T):
                    for j in range(i + 1, T):

                        flow_path = os.path.join(
                            slice_flow_dir,
                            f"flow_t{i}_t{j}.npz"
                        )

                        if os.path.isfile(flow_path):
                            continue

                        delta_t = max(times[j] - times[i], 1e-4)

                        # ---------- build c_i, c_j ----------
                        sd_i = sd_from_mask(masks_np[i])
                        sd_j = sd_from_mask(masks_np[j])

                        c_i = c_from_sd(sd_i)
                        c_j = c_from_sd(sd_j)

                        # ---------- RD-only fit (local, cheap) ----------
                        lap = laplacian_neumann(c_i)
                        dtc = (c_j - c_i) / delta_t
                        phi1 = lap
                        phi2 = c_i * (1 - c_i)

                        band = np.abs(sd_i) < 10.0
                        X = np.stack([phi1[band], phi2[band]], axis=1)
                        y = dtc[band]

                        if y.size < 10:
                            continue

                        coef, _ = nnls(X, y)
                        D_hat, rho_hat = coef

                        # ---------- RD forward ----------
                        c_RD = c_i.copy()
                        dt = delta_t / nsteps_rd
                        for _ in range(nsteps_rd):
                            c_RD = c_RD + dt * (
                                D_hat * laplacian_neumann(c_RD)
                                + rho_hat * c_RD * (1 - c_RD)
                            )
                            c_RD = np.clip(c_RD, 0, 1)

                        # ---------- residual SD-flow ----------
                        sd_RD = sd_from_c(c_RD)
                        sd_T  = sd_j

                        g0 = sd_proxy(sd_RD)
                        g1 = sd_proxy(sd_T)

                        v_back, u_back = optical_flow_tvl1(g1, g0)

                        vy = -v_back / delta_t * flow_scale
                        vx = -u_back / delta_t * flow_scale

                        vx = gaussian_filter(vx, vel_sigma)
                        vy = gaussian_filter(vy, vel_sigma)

                        # ---------- soft support ----------
                        w = soft_support(sd_i)
                        vx *= w
                        vy *= w

                        # ---------- save (same format) ----------
                        np.savez_compressed(
                            flow_path,
                            vx=vx.astype(np.float32),
                            vy=vy.astype(np.float32),
                            t0=times[i],
                            t1=times[j],
                        )

                self.flow_by_slice.append(slice_flow_dir)

                num_slices_curr_patient += 1
                curr_slice_idx += 1

            self.patient_id_to_slice_id.append(
                np.arange(curr_slice_idx - num_slices_curr_patient, curr_slice_idx)
            )

    def __len__(self):
        return len(self.all_patient_ids)
    
    def num_image_channel(self) -> int:
        return 1


class BrainUCSFMaskSubset(BrainUCSFMaskDataset):
    """
    Subset for PDF training / validation / testing.
    Returns (images, masks, timestamps).
    """

    def __init__(self,
                #  main_dataset: BrainUCSFMaskDataset,
                #  subset_indices: List[int],
                 main_dataset: BrainUCSFMaskDataset = None,
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

            for local_sid, global_sid in enumerate(slice_ids):
                self.image_by_slice.extend(
                    [main_dataset.image_by_slice[global_sid]]
                )
                self.mask_by_slice.extend(
                    [main_dataset.mask_by_slice[global_sid]]
                )
                self.time_by_slice.extend(
                    [main_dataset.time_by_slice[global_sid]]
                )
        
        self.return_format = return_format
                

    def __len__(self):
        return len(self.image_by_slice)

    def __getitem__(self, idx):
        """
        Returns:
            images: [2, 1, H, W]
            masks:  [2, 1, H, W]
            times:  [2]
        """

        img_paths = self.image_by_slice[idx]
        msk_paths = self.mask_by_slice[idx]
        times_all = self.time_by_slice[idx]
        
        # --------------------------------------------------
        # sample random pair (one_pair mode)
        # --------------------------------------------------

        pair_idx = np.random.choice(len(img_paths), size=2, replace=False)
        pair_idx = np.sort(pair_idx)
        i, j = pair_idx.tolist()

        # --------------------------------------------------
        # load images & masks
        # --------------------------------------------------
        images = []
        masks  = []

        for ip, mp in zip([img_paths[i], img_paths[j]],
                          [msk_paths[i], msk_paths[j]]):

            img = load_image(ip, target_dim=self.target_dim, normalize=False)
            img = normalize_image(img)
            img = add_channel_dim(img)

            msk = load_image(mp, target_dim=self.target_dim, normalize=False)
            msk = (msk > 0).astype(np.float32)   # whole tumor
            msk = add_channel_dim(msk)

            images.append(img)
            masks.append(msk)

        images = np.stack(images, axis=0)   # [2,1,H,W]
        masks  = np.stack(masks, axis=0)    # [2,1,H,W]
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
    dataset = BrainUCSFDataset(max_slice_per_patient=None)
    dataset.return_statistics()

    print('Subset with max 20 slices per patient.')
    dataset = BrainUCSFDataset(max_slice_per_patient=20)
    dataset.return_statistics()
