# ReferWhat
Source code and data for paper [Referring Expression Generation via Visual Dialogue](https://link.springer.com/chapter/10.1007/978-3-030-60457-8_3)
# Install requirment
* yaml
* NLTK
* h5py
* json
* TensorboardX
* Java1.8.0
* Python3.6
* PyTorch 1.1
# Dataset
* Image: Download from [COCO_train2014](http://images.cocodataset.org/zips/train2014.zip)
* ReferWhat?! append only dialog: Download from [ReferWhat_dataset](https://drive.google.com/file/d/1xQPscO1H2m1-Zb-jWzSx4o-AoyfLpG_e/view?usp=sharing), and save into `/data` folder.
# Prepare
`cd code/prepare`
* image:
`python extract_feats.py --dataset refcoco --cnn resnet152 --image_root path_fo_COCO_image --data_root ../../data/`
* dataset:
`python load_VD_data.py --path ../../data/refcoco/unc_split`
# Train
* Step1: modify the config file in code/config
* Step2: `python main.py`
# Evaluation
`python Gen_eval.py`
