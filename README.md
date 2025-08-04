# GAANet
GAANet official code， 
Since I am currently working, I will organize all the core code when I have free time later. Currently, the code is only for rotating object detection, and the core module architecture is missing and needs to be organized.

## Install 
**CUDA Driver Version ≥ CUDA Toolkit Version(runtime version) = torch.version.cuda**

a. Create a conda virtual environment and activate it, e.g.,
```
conda install pytorch==1.10.1 cudatoolkit==11.3.1 torchvision==0.11.2 -c pytorch
```

```
pip install -r requirements.txt
cd utils/nms_rotated
python setup.py develop  #or "pip install -v -e ."
```
## Install DOTA_devkit. 

### Download DOTA_devkit. 

-[DOTA_devkit]  [download](https://pan.baidu.com/s/1MBW3DK6Vjx09T5dJdiXnig) password:peoe

**(Custom Install, it's just a tool to split the high resolution image and evaluation the obb)**
```
cd GAANet/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```
## test last_predictions_Txt
**last_predictions_Txt** is our best prediction result on YOLO11 as the benchmark model. To get the detection accuracy score on DroneVehicle, you need to:
1. Modify the path in DOTA_devkit-master/dota_evaluation_task1.py
2. 
```
python DOTA_devkit-master/dota_evaluation_task1.py 
```
