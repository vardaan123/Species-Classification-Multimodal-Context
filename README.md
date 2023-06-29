# Species Classification using Multimodal Heterogeneous Context

We present a species classification model that is based on reformulation of image classification as link prediction in a multimodal KG. The multimodal knowledge graph may include diverse forms of heterogeneous contexts that pertain to different modalities, such as numerical information for locations and time, categorical data for species/taxon IDs, and visual content such as images.

## Installation

```
pip install -r requirements.txt
```

### Species classification on a single image input
Note: Some sample camera trap species images are available in the dir. `data/sample_images/`. The trained checkpoint is available in the dir. `ckpts/`

```
python eval_image.py --ckpt-path <PATH TO TRAINED CKPT> --img-path <PATH TO IMG FILE>
```

## Data Preprocessing
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
