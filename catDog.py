import sys
from PyQt5.QtWidgets import QApplication,QWidget,QFileDialog
from window import Ui_Dialog
from PyQt5.QtGui import QPixmap,QPainter,QColor,QPen
from ultralytics import YOLO#导入YOLO模型类

class catDogDetector:
    def __init__(self,model_path):
        self.model = YOLO(model_path)#加载传入的YOLO模型 从而进行后续的检测

    def detect(self,image_path):
        #使用YOLO模型 检测图像
        results = self.model(image_path)#对所选的图像进行检测
        boxes = results[0].boxes#获取边界框信息
        classes = boxes.cls.tolist()#获取标签
        xy = boxes.xyxy.tolist()#获取边界框坐标

        return classes,xy #返回检测的目标的类别标签

class mainWindow(QWidget,Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)#显示QT Designer中的控件布局内容
        self.setWindowTitle("猫狗分类识别系统")#设置窗口标题
        self.slot_init()
        self.detector = catDogDetector("./best.pt")
    def slot_init(self):
        self.btn_choose.clicked.connect(self.loadImage)#连接选择图片的按钮点击事件与槽函数
        self.btn_identify.clicked.connect(self.detectCatDog)#连接识别按钮与检测函数
    def loadImage(self):
        file_name = QFileDialog.getOpenFileName(self,"选择图片","","Images(*.png *.jpg *.xpm)")[0]#打开文件选择器并选择图片
        if file_name:
            self.label_image.setPixmap(QPixmap(file_name).scaled(self.size()))#在label控件里显示所选择的图片
            self.image_path = file_name#保存路径 给下一步使用

    def detectCatDog(self):
        if hasattr(self,'image_path'):#检测是否选择了图片
            result,xy = self.detector.detect(self.image_path)#调用检测函数
            classes = result#将检测结果转换为列表
            class_name = ["狗","猫"]#类别标签 0是狗 2是猫
            detect_classes = []#定义空的类别结果
            image = QPixmap(self.image_path)
            painter = QPainter(image)
            for i,cls in enumerate(classes):
                # enumerate() 是 Python 内置的一个函数，用于在遍历序列（如列表、元组、字符串）时，提供元素的索引和值。它可以让你在循环中同时访问元素和它们的索引，避免手动维护索引变量。
                box = xy[i]#获取对应边界框的坐标
                x1,y1,x2,y2 = map(int,box)
                if cls == 0:
                    detect_classes.append(class_name[0])#添加到 狗的类别结果里
                    color = QColor(255,0,0)#红色框
                elif cls == 2:
                    detect_classes.append(class_name[1])  # 添加到 猫的类别结果里
                    color = QColor(0, 0, 255)  # 蓝色框
                pen = QPen(color,3)
                painter.setPen(pen)
                painter.drawRect(x1,y1,x2,y2)
                painter.drawText(x1, y1 - 5, detect_classes[-1])

            painter.end()
            self.label_image.setPixmap(image.scaled(self.size()))#重新绘制带有检测框的图像

            if detect_classes:
                self.label_result.setText(f"识别结果: {', '.join(detect_classes)}")#显示结果
            else:
                self.label_result.setText("未识别到猫或狗")
        else:
            self.label_result.setText("未识别到猫或狗")



if __name__ == '__main__':
    app = QApplication(sys.argv)#创建应用程序实例
    window = mainWindow()#创建主窗口实例
    window.show()#显示窗口
    sys.exit(app.exec_())#启动事件循环