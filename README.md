建筑及道路-裂缝分割数据集   共4000张 目标检测yolo格式

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/36870f8bca4e435ca575ea2f2e035acf.png)

训练了yolov5n-seg，yolov5m-seg，yolov5x-seg三个模型，训练了100epoch![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7e849f6f6ecb496796474205ff685bb1.png)

训练任务为分割 segment![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/74ec5ed1c7374963adf7b5fdac2f0ceb.png)

训练数据集和检测结果如图，![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1c5c3a3fc2b74cdfb77160fc83e61d21.png)
4000张yolo格式的裂缝分割数据

# 混凝土裂缝分割项目（YOLOv5-seg）
## 项目简介
基于YOLOv5-seg实现混凝土裂缝实例分割，数据集共4000张YOLO格式标注图像；分别训练`yolov5n-seg`、`yolov5m-seg`、`yolov5x-seg`三个模型，训练轮次100 epoch，可完成裂缝像素级分割，输出裂缝掩码，用于建筑结构病害评估。

## 数据集信息表
|项目|详情|
| ---- | ---- |
|数据集名称|混凝土裂缝分割数据集|
|图像总数|4000张|
|标注格式|YOLO txt分割标注|
|类别数量|1类|
|训练集|3200张|
|验证集|480张|
|测试集|320张|

### 类别标签
|英文标签|中文标签|
| ---- | ---- |
|crack|混凝土裂缝|
一句话类别：仅包含混凝土裂缝1类目标的实例分割数据集。

**关键词**：混凝土裂缝、裂缝实例分割、YOLOv5-seg、建筑病害、结构检测、像素级分割

## 应用场景
1. 建筑、桥梁、隧道混凝土表面裂缝识别与像素级面积统计，结构安全评估。
2. 无人机/相机近距离采集混凝土表面图像，自动分割裂缝，替代人工目视测量。
3. 土木工程、机器视觉毕设/算法研究，支持对比n/m/x不同规模模型精度与推理速度。
4. 建筑运维系统，批量处理图片，自动计算裂缝长度、宽度、病害占比。

## 项目文件清单
- YOLOv5-seg
- 4000张裂缝数据集（images+labels）
- 训练权重：`yolov5n-seg.pt`、`yolov5m-seg.pt`、`yolov5x-seg.pt`
- 训练日志、PR曲线、预测结果可视化图

## 简易构建代码
### 1. 数据集配置文件 `crack_seg.yaml`
```yaml
path: ./crack_dataset
train: images/train
val: images/val
test: images/test

names:
  0: crack
```

### 2. 训练脚本 `train_crack_seg.py`
```python
# YOLOv5-seg 训练脚本
import torch
from utils.datasets import LoadImages
from models.yolo import Model

if __name__ == "__main__":
    # 选择模型：yolov5n-seg / yolov5m-seg / yolov5x-seg
    model_type = "yolov5m-seg"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 启动训练
    from train import run
    run(
        data="crack_seg.yaml",
        weights=f"{model_type}.pt",
        epochs=100,
        imgsz=640,
        batch=8,
        device=device,
        project="runs/train",
        name=model_type+"_crack",
        task="segment"
    )
```

### 3. 推理预测脚本 `predict_crack.py`
```python
from predict import run

if __name__ == "__main__":
    run(
        weights="runs/train/yolov5m-seg_crack/weights/best.pt",
        source="./test_img",
        imgsz=640,
        task="segment",
        save_mask=True, # 保存分割掩码
        save_txt=True
    )
```

### 环境安装
```bash
pip install torch torchvision opencv-python numpy matplotlib
```

## PyQt简易可视化界面代码
```python
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from predict import run

class CrackSegWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("混凝土裂缝分割检测系统")
        self.setGeometry(100, 100, 1000, 700)
        self.weight_path = "runs/train/yolov5m-seg_crack/weights/best.pt"
        self.init_ui()

    def init_ui(self):
        self.img_label = QLabel("图像预览区域", self)
        self.img_label.setGeometry(20, 20, 750, 600)
        self.img_label.setStyleSheet("border:1px solid #666;")
        self.img_label.setAlignment(Qt.AlignCenter)

        self.btn_detect = QPushButton("裂缝分割检测", self)
        self.btn_detect.setGeometry(800, 50, 160, 40)
        self.btn_detect.clicked.connect(self.detect_image)

    def detect_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择混凝土图片", "", "图片(*.jpg *.png *.bmp)")
        if not file_path:
            return
        run(weights=self.weight_path, source=file_path, imgsz=640, task="segment", save_mask=True)
        img_res = cv2.imread("runs/predict/exp/"+file_path.split("/")[-1])
        img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        h, w, c = img_res.shape
        qt_img = QImage(img_res.data, w, h, c*w, QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qt_img).scaled(self.img_label.size(), Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CrackSegWindow()
    win.show()
    sys.exit(app.exec_())
```


