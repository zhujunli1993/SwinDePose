# Pseudo-RGB FFB6D. 

## This is an implementation of FFB6D, with pseudo-RGB images as input.
To preview the MD file, press ctrl+k+v

### Run our FFB6D docker 
```bash 
sudo nvidia-docker run --gpus all --ipc=host --ulimit memlock=-1 --name 'new container name' -it --rm -v /raid/home/zl755:/workspace zhujunli/ffb6d:latest
```
### Generate Pseudo Images
It will generate both XYZ pseudo and signed angles pseudo images. 
```bash 
sh ffb6d/prepare_data/create_ycbv_pseudo.sh 
```

### Run Training Codes
* #### Trianing FFB6D with object masks: 
Pseudo-RGB image: 
[angle-X, angle-Y, angle-Z, signed_angle-X, signed_angle-Y, signed_angle-Z]. 
```bash 
sh scripts/train_ycb_mask.sh 
```
* #### Trianing FFB6D without object masks.
Pseudo-RGB image: 
[angle-X, angle-Y, angle-Z, signed_angle-X, signed_angle-Y, signed_angle-Z]. 
```bash 
sh scripts/train_ycb_full.sh 
```