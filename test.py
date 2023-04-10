import os
import cv2
import onnx
import time
import argparse
from onnxsim import simplify

import torch
from utils.tool import *
from module.detector import Detector



# 检查输出
def check_onnx_output(filename, input_data, torch_output):
    print("模型测试")
    session = onnxruntime.InferenceSession(filename)
    input_name = session.get_inputs()[0].name
    result = session.run([], {input_name: input_data.detach().cpu().numpy()})
    for test_result, gold_result in zip(result, torch_output.values()):
        np.testing.assert_almost_equal(
            gold_result.cpu().numpy(), test_result, decimal=3,
        )
    return result
# 检查模型
def check_onnx_model(model, onnx_filename, input_image):
    with torch.no_grad():
        torch_out = {"output": model(input_image)}
    check_onnx_output(onnx_filename, input_image, torch_out)
    print("模型输出一致")
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    print("模型测试成功")
    return onnx_model
if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default="", help='.yaml config')
    parser.add_argument('--weight', type=str, default=None, help='.weight config')
    parser.add_argument('--img', type=str, default='', help='The path of test image')
    parser.add_argument('--thresh', type=float, default=0.65, help='The path of test image')
    parser.add_argument('--onnx', action="store_true", default=False, help='Export onnx file')
    parser.add_argument('--torchscript', action="store_true", default=False, help='Export torchscript file')
    parser.add_argument('--cpu', action="store_true", default=False, help='Run on cpu')

    opt = parser.parse_args()
    assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"
    assert os.path.exists(opt.weight), "请指定正确的模型路径"
    assert os.path.exists(opt.img), "请指定正确的测试图像路径"

    # 选择推理后端
    if opt.cpu:
        print("run on cpu...")
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            print("run on gpu...")
            device = torch.device("cuda")
        else:
            print("run on cpu...")
            device = torch.device("cpu")     

    # 解析yaml配置文件
    cfg = LoadYaml(opt.yaml)    
    print(cfg) 

    # 模型加载
    print("load weight from:%s"%opt.weight)
    model = Detector(cfg.category_num, True).to(device)
    model.load_state_dict(torch.load(opt.weight, map_location=device))
    #sets the module in eval node
    model.eval()
    
    # 数据预处理
    ori_img = cv2.imread(opt.img)
    res_img = cv2.resize(ori_img, (cfg.input_width, cfg.input_height), interpolation = cv2.INTER_LINEAR) 
    img = res_img.reshape(1, cfg.input_height, cfg.input_width, 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0

    # 导出onnx模型
    if opt.onnx:
        torch.onnx.export(model,                     # model being run
                          img,                       # model input (or a tuple for multiple inputs)
                          "./FastestDet.onnx",       # where to save the model (can be a file or file-like object)
                          export_params=True,        # store the trained parameter weights inside the model file
                          opset_version=13,          # the ONNX version to export the model to
                          do_constant_folding=True)  # whether to execute constant folding for optimization
        # onnx-sim
        onnx_model = onnx.load("./FastestDet.onnx")  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        print("onnx sim sucess...")
        onnx.save(model_simp, "./FastestDet_sim.onnx")                  
         
    # 导出torchscript模型
    if opt.torchscript:
        import copy
        model_cpu = copy.deepcopy(model).cpu()
        x = torch.rand(1, 3, cfg.input_height, cfg.input_width)
        mod = torch.jit.trace(model_cpu, x)
        mod.save("./FastestDet.pt")
        print("to convert torchscript to pnnx/ncnn: ./pnnx FastestDet.pt inputshape=[1,3,%d,%d]" % (cfg.input_height, cfg.input_height))
    
    # 检测导出的onnx模型是否完整，输出是否一致
    onnx_name =  "FastestDet.onnx"
    check_onnx_model(model, onnx_name, img)
    # 模型推理
    start = time.perf_counter()
    preds = model(img)
    end = time.perf_counter()
    time = (end - start) * 1000.
    print("forward time:%fms"%time)

    # 特征图后处理
    output = handle_preds(preds, device, opt.thresh)

    # 加载label names
    LABEL_NAMES = []
    with open(cfg.names, 'r') as f:
	    for line in f.readlines():
	        LABEL_NAMES.append(line.strip())
    
    H, W, _ = ori_img.shape
    scale_h, scale_w = H / cfg.input_height, W / cfg.input_width

    # 绘制预测框
    for box in output[0]:
        print(box)
        box = box.tolist()
       
        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * W), int(box[1] * H)
        x2, y2 = int(box[2] * W), int(box[3] * H)

        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
        cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

    cv2.imwrite("result.png", ori_img)
