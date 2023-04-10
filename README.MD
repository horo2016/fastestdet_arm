
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

