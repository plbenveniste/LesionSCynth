from pathlib import Path
import torch
import torchio as tio
from torchio import SubjectsLoader, Subject
import lightning as L
import numpy as np
import warnings
from tqdm import tqdm
from typing import Optional, Iterator, List, Dict, Any, Union

from .data_augmentation import CarveMix
from ..configs.config import Config


def get_ext(subj_dir, filename):
    """ Use the uncompressed volume if it exists. """
    if (subj_dir / f'{filename}.nii').exists():
        return subj_dir / f'{filename}.nii'
    elif (subj_dir / f'{filename}.nii.gz').exists():
        return subj_dir / f'{filename}.nii.gz'
    else:
        return None


warnings.filterwarnings("ignore", ".*does not have many workers.*")


class BalancedSampler(torch.utils.data.RandomSampler):
    """ Over-samples underrepresented classes or under-samples over-represented classes in the dataset.
    The attribute to use for balancing should be present in all of the samples of the input data_source. You can simply
    apply over-sampling or under-sampling, independent of which class/group is more prevalent, by using sampling_type.
    Otherwise, if you want to take all the samples of a specific class/group and then either over-sample or under-sample
    the other group(s), then use balance_with. Finally, if a balance is not required, but rather a specific ratio,
    supply this with the input ratio. Exactly one of these three options should be supplied (sampling_type, balance_with
    or ratio).
    Args:
        data_source (tio.SubjectsDataset): The dataset to sample from.
        balance_attribute (str): The balance will be achieved with respect to this attribute. Each subject in the
                                 data_source should have this attribute.
        num_samples (int): Number of samples to draw. If None, the number of samples is equal to the number of samples
                           in the dataset.
        sampling_type (str): The type of sampling to perform. Either 'over' or 'under'.
        balance_with (int or str): The label value of balance_attribute to balance the dataset with. E.g. if 0,
            then the dataset will contain all samples with balance_attribute equal to 0, as well as an equal number
            of samples from the other categories.
        ratio (dict): If a specific ratio of samples between classes/groups is required, specify as dict, e.g.
                      {0: 3, 1: 1, 2: 1}, if there are three possible values in balance_attribute, and we want three
                      times as many of group 0 than group 1 or group 2.
        """
    def __init__(self, data_source: tio.SubjectsDataset, balance_attribute: str,
                 num_samples: Optional[int] = None, sampling_type: Optional[str] = None,
                 balance_with: Optional[Union[int, str]] = None,
                 ratio: Optional[Dict] = None,):
        self.balance_attribute = balance_attribute
        self.category_labels = [subj[self.balance_attribute] for subj in data_source]
        self.sampling_type = sampling_type
        self.balance_with = balance_with
        if ratio:
            if not all(isinstance(v, (int, float)) for v in ratio.values()):
                raise ValueError("All values in the ratio dict must be of type int or float.")
            else:
                total = sum(ratio.values())
            # Normalise so the entries sum to 1
            self.ratio = {k: v / total for k, v in ratio.items()}
        else:
            self.ratio = None

        self.sense_check()  # Sense-check the inputs

        super().__init__(data_source, replacement=False, num_samples=num_samples, generator=None)

    def sense_check(self):
        if self.sampling_type is not None and self.sampling_type not in ['over', 'under']:
            raise ValueError(f"Sampling type {self.sampling_type} not recognised.")
        if (self.sampling_type is None) == (self.balance_with is None):
            raise ValueError("Exactly one of sampling_type or balance_with must be provided.")

        if self.n_categories < 2:
            raise ValueError("Less than 2 categories present - cannot apply BalancedSampler")

        if self.balance_with is not None:
            if self.balance_with not in self.category_labels:
                raise ValueError(f"Using balance_with={self.balance_with} in BalancedSampler, but this value is not "
                                 f"present in the attribute {self.balance_attribute} in the data.")

        if self.ratio is not None:
            for cat in self.ratio.keys():
                if cat not in self.category_labels:
                    raise ValueError(f"Supplied category {cat} in ratio dict input to BalancedSampler but this value"
                                     f"is not present in the attribute {self.balance_attribute} in the data ")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return int(sum(self.n_per_category.values()))
        return self._num_samples

    @property
    def n_categories(self) -> int:
        return len(np.unique(self.category_labels))

    @property
    def n_per_category(self) -> Dict:
        unique_cats, cat_counts = np.unique(self.category_labels, return_counts=True)
        if self.balance_with is not None:
            base_n = cat_counts[unique_cats == self.balance_with][0]
            base_cat = self.balance_with
        elif self.sampling_type is not None:
            if self.sampling_type == 'over':
                base_n = np.max(cat_counts)
                base_cat = unique_cats[np.argmax(cat_counts)]
            elif self.sampling_type == 'under':
                base_n = np.min(cat_counts)
                base_cat = unique_cats[np.argmin(cat_counts)]
            else:
                raise ValueError(f"Sampling type {self.sampling_type} not recognised.")
        else:
            raise ValueError("Exactly one of sampling_type or balance_with must be provided.")

        if self.ratio:
            # Set the number of samples per category relative to the base category
            return {cat: round(base_n * r / self.ratio[base_cat]) for cat, r in self.ratio.items()}
        else:
            # Otherwise, the number of samples is the same for each category
            return {cat: base_n for cat in unique_cats}

    def get_balanced_indices(self) -> List:
        category_labels = self.category_labels
        unique_cats, cat_counts = np.unique(category_labels, return_counts=True)

        indices = []
        for cat, cnt in zip(unique_cats, cat_counts):
            # Get the indices of the subjects with the current category
            cat_indices = np.where(category_labels == cat)[0]
            if cnt == self.n_per_category[cat]:
                samples = cat_indices.tolist()
            elif cnt > self.n_per_category[cat]:
                # Undersample overrepresented categories
                samples = np.random.choice(cat_indices, self.n_per_category[cat], replace=True).tolist()
            else:
                # Oversample underrepresented categories
                # First repeat all elements an equal number of times
                samples = np.repeat(cat_indices, self.n_per_category[cat] // cnt).tolist()
                # Randomly sample the indices to fill any remaining spaces
                samples += np.random.choice(cat_indices, self.n_per_category[cat] % cnt).tolist()

            indices.extend(samples)

        return indices

    def __iter__(self) -> Iterator[int]:
        balanced_indices = self.get_balanced_indices()
        n = len(balanced_indices)

        for _ in range(self.num_samples // n):
            yield from np.random.permutation(balanced_indices)
        yield from np.random.choice(balanced_indices, self.num_samples % n)


class DataModule(L.LightningDataModule):
    def __init__(self, data_dir: Path, config: Config):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.train_set, self.val_set, self.test_set = None, None, None
        self.sampler = self.config.sampler
        self.training_transform = self.config.training_transform
        self.validation_transform = self.config.validation_transform

    def setup(self, stage: str):
        """ Set up the train/validation or test sets
        Args:
            stage: Either "fit" or "test"
        Returns: None, sets the train_set and val_set, or test_set, attributes
        """
        if stage == "fit":
            self.train_set = self.setup_dataset(self.data_dir, self.config.training_transform, subset='train',
                                                subdirs=self.config.train_dirs)
            self.val_set = self.setup_dataset(self.data_dir, self.config.validation_transform, subset='validation',
                                              subdirs=self.config.val_dirs)
        elif stage == "test":
            self.test_set = self.setup_dataset(self.data_dir, self.config.validation_transform, subset='test')
        else:
            raise ValueError(f"Stage {stage} not recognised.")
        return

    def train_dataloader(self):
        patches_training_set = tio.Queue(
            subjects_dataset=self.train_set,
            max_length=self.config.max_queue_length,
            samples_per_volume=self.config.samples_per_volume,
            sampler=self.sampler,
            num_workers=self.config.num_workers(),
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        return SubjectsLoader(patches_training_set, batch_size=self.config.training_batch_size, num_workers=0,
                                 pin_memory=self.config.pin_memory)

    def train_dataloader_balanced(self):
        print('Train Set:')
        labels = [subj.label for subj in self.train_set]
        print(f'Number of lesion subjects: {sum(labels)}')
        print(f'Number of non-lesion subjects: {len(labels) - sum(labels)}')

        # Balance epochs such that we take all subjects with real lesions and an equal number of non-lesion subjects
        subject_sampler = BalancedSampler(self.train_set, **self.config.sampler_args)

        patches_training_set = tio.Queue(
            subjects_dataset=self.train_set,
            max_length=self.config.max_queue_length,
            samples_per_volume=self.config.samples_per_volume,
            sampler=self.sampler,
            subject_sampler=subject_sampler,
            num_workers=self.config.num_workers(),
            shuffle_subjects=False,
            shuffle_patches=True,
        )
        return SubjectsLoader(patches_training_set, batch_size=self.config.training_batch_size, num_workers=0,
                                 pin_memory=self.config.pin_memory)

    def val_dataloader(self):
        patches_validation_set = tio.Queue(
            subjects_dataset=self.val_set,
            max_length=self.config.max_queue_length,
            samples_per_volume=self.config.samples_per_volume*2,
            sampler=self.sampler,
            num_workers=self.config.num_workers(),
            shuffle_subjects=False,
            shuffle_patches=False,
        )
        return SubjectsLoader(patches_validation_set, batch_size=self.config.validation_batch_size)

    def load_subject(self, subj_dir: Path) -> Union[tio.Subject, None]:
        filepath = get_ext(subj_dir, self.config.modalities[0])
        if filepath is None:
            # Skip this subject if file does not exist
            return
        images = {self.config.modalities[0]: tio.ScalarImage(filepath)}

        seg_path = get_ext(subj_dir, 'seg')
        sc_seg_path = get_ext(subj_dir, 't2_sc_seg')
        if seg_path and sc_seg_path:
            images['segmentation'] = tio.LabelMap(seg_path)
            # Determine class based on lesion/no lesion
            label = 1 if images['segmentation'][tio.DATA].sum() > 0 else 0
            # Load the SC mask and ensure in same space as the lesion segmentation mask
            images['sc_seg'] = tio.LabelMap(sc_seg_path)
            images['sc_seg'] = tio.Resample(images['segmentation'])(images['sc_seg'])
        else:
            return

        return Subject(images, name=subj_dir.name, label=label)

    def get_all_subjects(self, dirpath: Path, subdirs: Optional[list] = None):
        subjects = []
        # subdirs could be just ['train'] or could be ['train-t2stir', 'train-t2', 'train-stir'] for example
        subdirs = [''] if subdirs is None else subdirs
        for subdir in subdirs:
            total = len(list((dirpath / subdir).iterdir()))
            for subj in tqdm((dirpath / subdir).iterdir(), total=total):
                if subj.is_dir():
                    subject = self.load_subject(subj)
                    if subject is None:
                        continue
                    try:
                        subject.check_consistent_space()
                    except RuntimeError as e:
                        print(f"Error in subject {subj.name}: {e}")
                        continue

                    subjects.append(subject)

        return subjects

    def setup_dataset(self, dirpath: Path, transform, subset: str, subdirs: Optional[list] = None):
        # Ensure that config.modalities has type list, and if not, convert it
        if not isinstance(self.config.modalities, list):
            self.config.modalities = [self.config.modalities]

        subjects = self.get_all_subjects(dirpath, subdirs)
        subjects = self.subjects_subset(subjects, subset)

        subdirs_msg = f'\nSub-directories: {subdirs}' if subdirs else ''
        print(f'Data Module:\nPath: {self.data_dir}{subdirs_msg}\nLoaded {len(subjects)} subjects\n')
        return tio.SubjectsDataset(subjects, transform=transform)

    def subjects_subset(self, subjects: List[Subject], subset: str) -> List[Subject]:
        # By default, return all subjects
        return subjects


class LesionsOnlyDataModule(DataModule):
    def subjects_subset(self, subjects: List[Subject], subset: str) -> List[Subject]:
        if subset == 'train':
            # Take only images with lesions
            return [subj for subj in subjects if subj.label == 1]

        return subjects


class SyntheticMixedDataModule(DataModule):
    def train_dataloader(self):
        return self.train_dataloader_balanced()


class CarveMixDataModule(SyntheticMixedDataModule):
    def setup_dataset(self, dirpath: Path, transform, subset: str, subdirs: Optional[list] = None):
        # Ensure that config.modalities has type list, and if not, convert it
        if not isinstance(self.config.modalities, list):
            self.config.modalities = [self.config.modalities]

        subjects = self.get_all_subjects(dirpath, subdirs)
        subjects = self.subjects_subset(subjects, subset)

        subdirs_msg = f'\nSub-directories: {subdirs}' if subdirs else ''
        print(f'Data Module:\nPath: {self.data_dir}{subdirs_msg}\nLoaded {len(subjects)} subjects\n')

        # Create the augmentation transform, supplying the list of subjects with lesions
        lesion_subjects = [subj for subj in subjects if subj.label == 1]
        carvemix_transform = CarveMix(lesion_subjects, im_name='t2', seg_name='segmentation')
        transform = tio.Compose([carvemix_transform, transform])

        return tio.SubjectsDataset(subjects, transform=transform)




