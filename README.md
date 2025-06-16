# LesionSCynth
This repository contains code for generating synthetic lesions in spinal cord MRI.

### Synthesis Examples
The LesionSCynth class for synthesis is in `model_and_training/data_augmentation.py` and a 'light' version of the requirements can 
be installed if just running this script (but not really light as PyTorch is required)

Plot example of synthetic lesions using the sample data:
```bash
conda create -n lesion_synth python=3.11
conda activate lesion_synth
pip install -r requirements_lite.txt
python -m LesionSCynth.model_and_training.data_augmentation \
  --lesion_dir data/preprocessed/lesions \
  --example_im_path data/preprocessed/images/sub-32vuiisIngenia06/t2.nii.gz \
  --sc_seg_path data/preprocessed/images/sub-32vuiisIngenia06/t2_sc_seg.nii.gz
```

### Training
If wanting to run the full training pipeline, you can install the full requirements `requirements.txt`. There is a further dependency on spinal cord toolbox (SCT) for obtaining segmentation masks of the spinal cord during preprocessing.

The preprocessing (& training) assume a particular directory structure and file naming. The directory structure 
for training & evaluation will be created by the preprocessing script.
```
|-- /path/to/rawdata
    |-- sub-1
        |-- t2.nii.gz
        |-- seg.nii.gz   # Binary lesion segmentation mask
    |-- sub-2
        |-- t2.nii.gz
        |-- seg.nii.gz
    .....
```

#### 1. Calculate the intensity statistics
We compute the contrast between the lesions in the dataset and their surrounding tissue, and then use the computed statistics to parametrise a distribution for synthesis.

The following requires spinal cord segmentation masks. If these are not already available, then run the preprocessing step first, which will generate these masks. 
Then use the arguments `--sc_seg_dir` and `--sc_seg_name` to specify the paths when calling the `intensity_stats` script.
E.g. `--sc_seg_dir data/preprocessed/images` and `--sc_seg_name intermediate_files/t2_sc_seg_processed.nii.gz`.

```bash
python -m LesionSCynth.preprocessing.intensity_stats \
  --anat_dir data/rawdata \
  --out_dir data/intensity_stats \
  --save_cc \
  --save_dilations \
  --sc_seg_dir data/preprocessed/images \
  --sc_seg_name intermediate_files/t2_sc_seg_processed.nii.gz
```

This gives us the statistics for each lesion saved in `data/intensity_stats/lesion_intensity_stats_[timestamp].csv`.


#### 2. Preprocess the data

`--metadata_path` is a CSV file with columns `id` (matching the subject or volume ID) and `section` (cerv or thor). This determines which spinal cord segmentation processing will be used for the acquisition.
```bash
python -m LesionSCynth.preprocessing.preprocess \
  --data_dir data/rawdata/ \
  --output_dir data/preprocessed/images \
  --metadata_path data/section_metadata.csv  \
  --lesion_outdir data/preprocessed/lesions/all  # to extract all lesions from the raw data
```

#### 3. Train the model
First, set the paths in the base config file (`configs/base_config.py`), e.g.,
```python
data_dir = 'data/preprocessed/images'  # The root directory for the training data
contrast_summary_path = 'data/intensity_stats/lesion_intensity_stats_20250526154909.csv'  # Path to the CSV file containing the contrast summary stats
save_examples_dir = None  # Directory to save training examples after augmentation
lesion_dir = 'data/preprocessed/lesions'  # Directory containing the lesion masks (& maybe intensity images). subdir defined in training_dirs_lesions
training_dirs = ['training_example']
```

Then launch a training with the below. Note the dot notation for the config path in the below.
```bash
python -m LesionSCynth.model_and_training.train \
  --config LesionSCynth.configs.mixed_synthetic_Adam \
  --out_dir data/models/
```

#### 4. Evaluation 

##### Launch inference
```bash
python -m LesionSCynth.testing.inference \
--input_dir data/preprocessed/images \
--orig_dir data/rawdata \
--preds_dir data/preds \
--model_dir data/models \
--path_to_bounds data/preprocessed/images/bounds_20250616152016.json 
```

##### Launch evaluation
The process and code from MS-Multi-Spine-Challenge were used for evaluation. See https://gitlab.inria.fr/msmultispinechallenge/msmultispineevaluation

```bash
python -m LesionSCynth.testing.run_froc \
--eval_dir data/eval/best_loss \
--preds_dir data/preds/best_loss \
--froc_script_path /path/to/main.py  # Path to the script from MS-Multi-Spine-Challenge evaluation 
```


##### Data Source
The two image examples used for example purposes in this repo are from:\
Cohen-Adad, J., Alonso-Ortiz, E., Abramovic, M., et al. (2021). Open-access quantitative MRI data of the spinal cord and reproducibility across participants, sites and manufacturers. Scientific Data, 8(1). https://doi.org/10.1038/s41597-021-00941-8
