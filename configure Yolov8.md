<aside>
💡

### 相关概念：

1. **卷积层（Convolution Layer）**：
    - 卷积操作通过卷积核（filter）提取输入数据的局部特征。
    - 卷积核在输入数据上滑动，计算点积，生成特征图（Feature Map）。
    - 卷积层能够捕捉图像的边缘、纹理等低级特征。
2. **池化层（Pooling Layer）**：
    - 用于下采样特征图，减少数据维度，同时保留重要信息。
    - 常见的池化方法有最大池化（Max Pooling）和平均池化（Average Pooling）。
3. **激活函数（Activation Function）**：
    - 常用 ReLU（Rectified Linear Unit）激活函数引入非线性，使模型能够学习复杂特征。
4. **全连接层（Fully Connected Layer）**：
    - 将卷积层和池化层提取的特征映射到输出空间，用于分类或回归任务。
5. **归一化（Normalization）**：
    - 批归一化（Batch Normalization）可以加速训练并提高模型的稳定性。

### 工作流程

1. 输入图像（如 28x28 的灰度图像）。
2. 通过卷积层提取局部特征（如边缘、纹理）。
3. 通过池化层下采样，减少特征图的大小。
4. 多次堆叠卷积层和池化层，提取更高级的特征。
5. 将特征输入全连接层，输出分类结果。

### 特点：

- **局部连接**：卷积操作只关注局部区域，减少计算量。
- **权值共享**：卷积核的参数在整个输入上共享，降低模型复杂度。
- **平移不变性**：对输入的平移具有鲁棒性。
</aside>

## Anaconda安装Pytorch和tensorflow

- 命令行，不解释了

## 安装YOLO

我安牛魔，cnm！，怎么又失败：

[windows下配置pytorch + yolov8+vscode，并自定义数据进行训练、摄像头实时预测_pytorch vscode-CSDN博客](https://blog.csdn.net/xwb_12340/article/details/131718725)

<aside>
💡

## 安装labellmg

1. **图像标注**：
    - 手动在图像上绘制边界框（Bounding Box），标注目标的位置和类别。
2. **支持多种标注格式**：
    - **Pascal VOC**：生成 `.xml` 文件，适用于许多深度学习框架。
    - **YOLO**：生成 `.txt` 文件，包含目标的类别和边界框坐标。
3. **多平台支持**：
    - 可在 Windows、macOS 和 Linux 上运行。
4. **简单易用**：
    - 提供图形化界面，用户可以通过鼠标操作完成标注。
</aside>

<aside>
💡

## labellmg打标签教程：

1. Github下载好软件：
2. 安装方式： 
    
    [Windows下深度学习标注工具LabelImg安装和使用指南 - Chen洋 - 博客园](https://www.cnblogs.com/cy0628/p/15581649.html)
    
3. 打开Anaconda prompt ，并进入安装目录：`(1)F：，(2) cd F:\Code-Numpy\labelImg\labelImg`
4. 输入：`pyrcc5 -o resources.py resources.qrc` 
`python labelImg.py`
5. 选yolo，划区域，打标签
6. 标签解释：分别是，对象类别，边界框的中心X坐标，边界框中心的y坐标，边界框的宽度，边界框的高度，一般归一化[0-1]

## 二、使用 LabelImg 标注图片

### 1. 打开图片或图片文件夹

- 启动后，点击“Open Dir”选择你的图片文件夹，软件会加载该文件夹下所有图片。
- 也可以用“Open”按钮打开单张图片。

### 2. 设置类别（标签）

- 在左侧“Label”输入框中输入类别标签（如“cat”、“dog”），按 Enter 添加。
- 也可以在“Label”列表中选择已有的类别。

### 3. 标注边界框

- 点击“Create RectBox”按钮（或按快捷键 `w`）。
- 在图片上拖动鼠标，绘制一个矩形框，框中即为目标的边界。
- 绘制完毕后，会弹出对话框要求输入该边界的类别标签。
- 输入类别后，点击“OK”。

### 4. 编辑和删除标注

- 选中某个边界框：点击该框。
- 删除：选中框后，按 `Delete` 键。
- 调整：可以拖动边界框到合适位置。

### 5. 保存标注

- 点击“Save”按钮（或快捷键 `Ctrl+S`）。
- 保存的文件格式有两种：
    - **Pascal VOC（.xml）**：常用于训练目标检测模型。
    - **YOLO（.txt）**：适用于YOLO系列模型。
- 可以在“Save Format”中选择你需要的格式
</aside>

## 配置Yolov8

## 二、使用 LabelImg 标注图片

### 1. 打开图片或图片文件夹

- 启动后，点击“Open Dir”选择你的图片文件夹，软件会加载该文件夹下所有图片。
- 也可以用“Open”按钮打开单张图片。

### 2. 设置类别（标签）

- 在左侧“Label”输入框中输入类别标签（如“cat”、“dog”），按 Enter 添加。
- 也可以在“Label”列表中选择已有的类别。

### 3. 标注边界框

- 点击“Create RectBox”按钮（或按快捷键 `w`）。
- 在图片上拖动鼠标，绘制一个矩形框，框中即为目标的边界。
- 绘制完毕后，会弹出对话框要求输入该边界的类别标签。
- 输入类别后，点击“OK”。

### 4. 编辑和删除标注

- 选中某个边界框：点击该框。
- 删除：选中框后，按 `Delete` 键。
- 调整：可以拖动边界框到合适位置。

### 5. 保存标注

- 点击“Save”按钮（或快捷键 `Ctrl+S`）。
- 保存的文件格式有两种：
    - **Pascal VOC（.xml）**：常用于训练目标检测模型。
    - **YOLO（.txt）**：适用于YOLO系列模型。
- 可以在“Save Format”中选择你需要的格式

## 三、配置yolov8：

1. 找到uartly文件路径：C:\Users\Admin\.conda\envs\newenv\Lib\site-packages\ultralytics
2. train 创建labels 和images;
3. 放入图片到labels ，然后放入数据
4. 然后进入val 同样创建labels 和images

### 四、训练自己的数据集

### 1. **准备数据集**

- **数据集结构**：YOLOv8要求数据集符合特定格式，通常使用COCO或YOLO格式。推荐使用YOLO格式，目录结构如下：
    
    text
    
    复制
    
    `dataset/
    ├── images/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   └── val/
    │       ├── image3.jpg
    ├── labels/
    │   ├── train/
    │   │   ├── image1.txt
    │   │   ├── image2.txt
    │   └── val/
    │       ├── image3.txt`
    
    - **images/**：存放训练和验证的图片。
    - **labels/**：存放标注文件，每个.txt文件与对应图片同名，格式为：
        
        text
        
        复制
        
        `<class_id> <x_center> <y_center> <width> <height>`
        
        - <class_id>：类别ID（从0开始）。
        - <x_center>, <y_center>, <width>, <height>：归一化后的边界框坐标（相对于图片宽高）。
        - 示例：
            
            text
            
            复制
            
            `0 0.5 0.5 0.2 0.3`
            
- **标注工具**：可以使用LabelImg、Roboflow或CVAT等工具生成标注文件。
- **划分数据集**：将数据集分为训练集（train）和验证集（val），通常按8:2或7:3的比例。

---

### 2. **创建数据集配置文件**

- 创建一个data.yaml文件，指定数据集路径和类别信息，内容如下：
    
    yaml
    
    复制
    
    `train: ./dataset/images/train  *# 训练集图片路径*
    val: ./dataset/images/val      *# 验证集图片路径*
    nc: 2                          *# 类别数量*
    names: ['cat', 'dog']          *# 类别名称*`
    
    - nc：类别数量。
    - names：类别名称列表，按class_id顺序。

---

### 3. **选择预训练模型**

- YOLOv8提供多种预训练模型（如yolov8n.pt、yolov8s.pt、yolov8m.pt等），根据需求选择：
    - yolov8n.pt：轻量，适合资源有限的设备。
    - yolov8m.pt：平衡性能和速度。
    - 下载预训练模型：运行yolo detect train时会自动下载，或从Ultralytics GitHub获取。

---

### 4. **训练模型**

- 使用以下命令启动训练：
    
    bash
    
    复制
    
    `yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16 device=0`
    
    - **参数说明**：
        - data：数据集配置文件路径（如data.yaml）。
        - model：预训练模型路径（如yolov8n.pt）。
        - epochs：训练轮数，建议50-100，视数据集大小调整。
        - imgsz：输入图片尺寸，通常为640x640。
        - batch：批次大小，根据GPU内存调整（如16、32）。
        - device：设备（0为GPU，cpu为CPU）。
    - **其他可选参数**：
        - patience：早停机制，防止过拟合（如patience=50）。
        - lr0：初始学习率（如lr0=0.01）。
        - augment：启用数据增强（默认开启）。
- 训练过程：
    - 训练日志会显示损失（box_loss、cls_loss、dfl_loss）和mAP指标。
    - 模型权重保存在runs/detect/trainX/weights/，包括best.pt（最佳模型）和last.pt（最新模型）。

---

### 5. **验证模型**

- 训练完成后，使用以下命令验证模型性能：
    
    bash
    
    复制
    
    `yolo detect val data=data.yaml model=runs/detect/trainX/weights/best.pt`
    
    - 输出包括mAP@50、mAP@50:95等指标。

---

### 6. **测试与推理**

- 使用训练好的模型进行推理：
    
    bash
    
    复制
    
    `yolo detect predict model=runs/detect/trainX/weights/best.pt source=your_image.jpg`
    
    - source：可以是图片、视频或文件夹路径。
    - 结果保存在runs/detect/predictX/。

---

### 7. **常见问题与优化**

- **数据不足**：使用数据增强（augment=True）或收集更多数据。
- **过拟合**：减少epochs或增加patience。
- **类别不平衡**：调整data.yaml中的类别权重，或使用Roboflow平衡数据集。
- **硬件限制**：降低batch或imgsz，或使用更轻量模型（如yolov8n）。
- **调试**：检查runs/detect/trainX/中的训练曲线（results.png）和混淆矩阵。

---

### 8. **工具与资源**

- **Roboflow**：用于数据集管理、预处理和增强。
- **Ultralytics文档**：查阅官方文档（[https://docs.ultralytics.com/）。](https://docs.ultralytics.com/%EF%BC%89%E3%80%82)
- **预训练模型**：从Ultralytics GitHub或Hugging Face下载。
- **可视化**：训练后查看runs/detect/trainX/中的结果图表。
