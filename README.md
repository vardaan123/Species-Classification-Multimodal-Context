# Species Classification using Multimodal Heterogeneous Context

We present a species classification model that is based on reformulation of image classification as link prediction in a multimodal KG. The multimodal knowledge graph may include diverse forms of heterogeneous contexts that pertain to different modalities, such as numerical information for locations and time, categorical data for species/taxon IDs, and visual content such as images.

## Installation

```
pip install -r requirements.txt
```

### Species classification on a single image input
Note: Some sample camera trap species images are available in the dir. `data/sample_images/`.

1. Download the trained checkpoint from [here](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/EQMx5KOJledHkhObXTemw3sBIJGQD4B_mvmPbri6-NE-lQ?e=E96LiI) and place it in the dir. `ckpts/`.
2. Download the data file from [here](https://buckeyemailosu-my.sharepoint.com/:x:/g/personal/pahuja_9_buckeyemail_osu_edu/Eca8a9n25adMt3EV0icU9CMB_SYDF89HvVi3dnC21iZA2w?e=cUOLu5) and place it in the dir. `data/iwildcam_v2.0/`.
3. Evaluate the pretrained model on a given image.
```
python eval_image.py --ckpt-path <PATH TO TRAINED CKPT> --img-path <PATH TO IMG FILE>
```

## Data Preprocessing
This will download the iWildCam2020-WILDS dataset and Open Tree of Life taxonomy and pre-process them.
```
bash preprocess.sh
```

Note: The dir. `data/iwildcam_v2.0/train/` contains images for all splits.

## Training a model from scratch

We consider different training settings that comprise of combination of different context types such as taxonomy, location, and time.

### Train using only images linked to species labels
```
python -u main.py --data-dir data/iwildcam_v2.0/ --img-dir data/iwildcam_v2.0/train/ --save-dir CKPT_DIR > CKPT_DIR/log.txt
```

### Train using species labels and taxonomy contexts
```
python -u main.py --data-dir data/iwildcam_v2.0/ --img-dir data/iwildcam_v2.0/train/ --save-dir CKPT_DIR --add-id-id > CKPT_DIR/log.txt
```

### Train using species labels and location contexts
```
python -u main.py --data-dir data/iwildcam_v2.0/ --img-dir data/iwildcam_v2.0/train/ --save-dir CKPT_DIR --add-image-location > CKPT_DIR/log.txt
```

### Train using species labels and time contexts
```
python -u main.py --data-dir data/iwildcam_v2.0/ --img-dir data/iwildcam_v2.0/train/ --save-dir CKPT_DIR --add-image-time > CKPT_DIR/log.txt
```

### Train using species labels, taxonomy, and time contexts
```
python -u main.py --data-dir data/iwildcam_v2.0/ --img-dir data/iwildcam_v2.0/train/ --save-dir CKPT_DIR --add-id-id --add-image-time > CKPT_DIR/log.txt
```

###  Train using species labels, taxonomy, and location contexts
```
python -u main.py --data-dir data/iwildcam_v2.0/ --img-dir data/iwildcam_v2.0/train/ --save-dir CKPT_DIR --add-id-id --add-image-location > CKPT_DIR/log.txt
```

### evaluate a model (specify split=val/test/id_val/id_test)
```
python eval.py --ckpt-path <PATH TO TRAINED CKPT> --split test --data-dir data/iwildcam_v2.0/ --img-dir data/iwildcam_v2.0/train/
```
