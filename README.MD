
copy  from https://github.com/dog-qiuqiu/FastestDet

fastestdet object detection 
目标检测


# train  
python3 train.py --yaml datasets/ts.yaml

# Export onnx   
You can export .onnx by adding the --onnx option when executing test.py
python3 test.py --yaml datasets/ts.yaml --weight weights/weight_AP05xx.pth --img w.jpg --onnx

# cd example test
cv-dnn
onnx-runtime

'orangepi@orangepizero2:~/yolo/FastestDet-main/example/onnx-runtime$ python3 runtime.py 
(9, 22, 22)
22
22
forward time:183.014402ms
=================box info===================
[175.0, 115.0, 213.0, 165.0, 0.9381985450176394, 3.0]  

[296.0, 112.0, 333.0, 163.0, 0.9314042658765402, 3.0]  

[230.0, 184.0, 269.0, 238.0, 0.7334719127275763, 3.0]  

[234.0, 221.0, 269.0, 271.0, 0.6670236771613627, 2.0]   



orangepi@orangepizero2:~/yolo/FastestDet-main/example/cv-dnn$ python3 main_lide.py   
Inference time: 130.41 ms


'
