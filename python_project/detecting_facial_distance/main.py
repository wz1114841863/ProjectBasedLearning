import sys
import time
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from client_1 import Ui_Form
from client_2 import Ui_Dialog
from client_3 import Ui_Dialog as UI_Dialog_Hobby
from img_select import imgSelect
from PySide6.QtCore import QTimer, QDateTime
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QDialog, QLabel
from PySide6.QtCore import QObject, QEvent, Signal, Slot, QThread


class WorkerThread(QThread):
    """
        定义线程类, 由mainWindow类进行创建和销毁
        调用摄像头,测量人脸距离
    """
    distance_changed = Signal(float)  # 使用Signal定义一个信号

    def __init__(self, img_select):
        super().__init__()
        self.is_interrupted = False
        self.distance = 0.0  # 人脸距离
        self.cap = cv2.VideoCapture(0)  # 调用摄像头
        self.detector = FaceMeshDetector(maxFaces=1)  # 人脸检测器
        self.sub_window_instance = None  # 子窗口
        self.img_select = img_select

    def run(self):
        while not self.is_interrupted:
            success, img = self.cap.read()  # 读取视频流
            img, faces = self.detector.findFaceMesh(img, draw=False)  # 检测人脸
            if faces:  # 检测到摄像头前有人脸
                face = faces[0]  # 先进行第一阶段
                pointLeft = face[145]  # 查找左眼，左边的值基本上是值145
                pointRight = face[374]  # 查找右眼，右边的值基本上是值374

                w, _ = self.detector.findDistance(
                    pointLeft, pointRight)  # 计算眼睛之间的距离
                """
                    f = (w * d) / W
                    定义宏值, 因为在只有一个摄像头的情况根据小孔成像原理. 需要知道摄像头的焦距f、
                    识别物体的实际宽度和成像后的物体的像素宽度才能推出物体距离摄像头的距离,所以
                    这里设置了两眼之间的宏值和摄像头的焦距, 根据求得的像素宽度w来计算距离
                """
                W = 6.3  # 人眼的左眼与右眼之间的距离为63mm，男人的为64mm，女人的为62mm，取中间值为63mm
                f = 6    # 电脑摄像头的焦距
                # 计算距离，TODO 需要考虑像素点和摄像头之间的转换关系
                self.distance = (W * f) / w * 100
                self.distance_changed.emit(self.distance)
                # 绘制线条，仅用于调试程序
            #     cv2.line(img, pointLeft, pointRight,
            #              (0, 200, 0), 3)  # 在眼睛之间绘制一条线
            #     cv2.circle(img, pointLeft, 5, (255, 0, 255),
            #                cv2.FILLED)  # 标注左眼
            #     cv2.circle(img, pointRight, 5,
            #                (255, 0, 255), cv2.FILLED)  # 标注右眼
            #     cvzone.putTextRect(  # 以字符串的方式显示距离
            #         img, f'Depth:{int(self.distance)}cm', (face[10][0]-95, face[10][1]-5), scale=1.8)

            # cv2.imshow("Iamge", img)
            # cv2.waitKey(1)

    def stop(self):
        self.is_interrupted = True


class subWindow(QDialog, Ui_Dialog):
    """
        弹出的子窗口
    """

    def __init__(self, img_select: imgSelect):
        super(subWindow, self).__init__()   # 调用父类 QDialog 的初始化方法
        self.setupUi(self)                  # 加载面布局和组件到当前对话框窗口
        self.img_select = img_select
        self.use_time = 0.0                 # 上一张图片反应的时间
        self.prev_img_id = -1               # 上一张图片的类别
        self.createItems()                  # 创建页面组件

    def createItems(self):
        self.timer = QTimer(self)             # 用于定时更新界面上的时间显示
        self.img_select.weight_init()
        img_path, idx = self.img_select.get_one_img()  # 获取图片和对应的种类
        self.prev_img_id = idx
        # print(self.img_select.subdirs_weight)
        image = QImage(img_path)           # 读取和展示图片
        pixmap = QPixmap.fromImage(image)
        self.label_pic.setPixmap(pixmap)
        self.label_pic.setScaledContents(True)
        self.label_pic.installEventFilter(self)
        self.label_warning.setText("请您注意坐姿，保护眼睛")
        self.timer.timeout.connect(self.updateTime)
        self.startTime = QDateTime.currentDateTime()
        self.timer.start(100)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """
            事件过滤器, 为label添加点击事件
            留作接口, 可以为label添加点击动作
        """
        if isinstance(obj, QLabel) and event.type() == QEvent.MouseButtonPress:
            pass
            return True  # 返回True表示事件被过滤器接收和处理
        return super().eventFilter(obj, event)

    def updateTime(self):                                   # 更新时间显示
        currentTime = QDateTime.currentDateTime()           # 获取当前时间
        elapsedTime = self.startTime.secsTo(currentTime)    # 计算时间差
        formatTime = self.formatTime(elapsedTime)           # 格式化并显示
        self.label_time.setText(f"持续时间: {formatTime}")

    def formatTime(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"

    def update_custom_hobby(self):
        """
            在子窗口关闭前记录用户的反应时间和图片的种类
            用于更新权重
        """
        currentTime = QDateTime.currentDateTime()           # 获取当前时间
        elapsedTime = self.startTime.msecsTo(currentTime)    # 计算时间差
        self.img_select.update_weight_by_time(elapsedTime, self.prev_img_id)


class subWindowHobby(QDialog, UI_Dialog_Hobby):
    """
        用户偏好选择测试性窗口
    """
    closeSignal = Signal()  # 自定义信号

    def __init__(self, img_select: imgSelect):
        super(subWindowHobby, self).__init__()
        self.counter = 0
        self.img_select = img_select        # 和父类共用一个imgSelect类对象
        self.setupUi(self)                  # 加载面布局和组件到当前对话框窗口
        self.createItems()                  # 创建页面组件
        self.use_time = []                  # 记录用户的平均反应时长, 未使用
        self.closeSignal.connect(self.closeIfReached)  # 关联自定义信号和槽函数

    def updateItems(self):
        """更新图片, 为每个label添加图片、事件过滤器和path_id属性"""
        paths, path_id = self.img_select.get_img_path()
        for i, path in enumerate(paths):
            image = QImage(path[0])
            pixmap = QPixmap.fromImage(image)
            # 根据索引动态获取对应的 label_pic_n 对象
            label = getattr(self, f"label_pic_{i+1}")
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            label.installEventFilter(self)
            label.setProperty('path_id', path_id[i])

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """事件过滤器, 为label添加点击事件"""
        if isinstance(obj, QLabel) and event.type() == QEvent.MouseButtonPress:
            # print("Label_path_id:", obj.property("path_id"))
            label = obj.property("path_id")
            self.img_select.update_weight(label)
            self.updateItems()
            self.counter += 1
            self.calcTime()
            self.updateShowCounter()
            # 计数超过10，关闭窗口
            if self.counter >= 10:
                self.closeSignal.emit()
            return True  # 返回True表示事件被过滤器接收和处理
        return super().eventFilter(obj, event)

    def getCounter(self):
        """获取计数次数"""
        return self.counter

    def updateShowCounter(self):
        """更新计数显示"""
        self.label_counter.setText(str(10 - self.getCounter()))

    def createItems(self):
        """初始化页面和定时器"""
        self.updateItems()
        self.updateShowCounter()
        # self.label_counter.connect(self.updateShowCounter)
        self.timer = QTimer(self)
        self.startTime = QDateTime.currentDateTime()
        self.timer.start()

    def calcTime(self):                                     # 更新时间显示
        """记录每次点击图片时, 所用的累积时长"""
        currentTime = QDateTime.currentDateTime()           # 获取当前时间
        elapsedTime = self.startTime.secsTo(currentTime)    # 计算与窗口打开时的时间差
        self.use_time.append(elapsedTime)

    @Slot()
    def closeIfReached(self):
        """关闭窗口"""
        self.close()


class mainWindow(QMainWindow, Ui_Form):
    def __init__(self, imgs_dir_root_path):
        super(mainWindow, self).__init__()  # 调用父类 QMainWindow 的初始化方法
        self.setupUi(self)                  # 用于设置界面的布局和组件

        self.img_select = imgSelect(imgs_dir_root_path)  # 实例化对象，用于和子窗口交互
        self.sub_window_instance = None     # 子窗口
        self.sub_window_hobby_inst = None   # 偏好选择窗口

        self.thread = None                  # 子线程
        self.createSignalSlot()             # 连接信号与槽

    def createSignalSlot(self):             # 与对应函数绑定
        self.pushButton_start_prog.clicked.connect(self.start_thread)
        self.pushButton_stop_prog.clicked.connect(self.stop_thread)
        self.pushButton_start_test.clicked.connect(self.startTest)
        self.pushButton_clear.clicked.connect(self.clearResult)
        self.pushButton_stop_prog.setEnabled(False)

    def start_thread(self):
        """
            启动子线程，调用摄像头, 获取距离
        """
        # 启动程序时关闭出结束程序之外的其他按键
        self.pushButton_start_prog.setEnabled(False)
        self.pushButton_stop_prog.setEnabled(True)
        self.pushButton_start_test.setEnabled(False)
        self.pushButton_clear.setEnabled(False)
        # 检查线程是否已经存在且正在运行
        if self.thread is None or not self.thread.isRunning():
            # 创建新的线程实例
            self.thread = WorkerThread(self.img_select)
            self.thread.start()
            self.thread.distance_changed.connect(
                self.changeDistance)  # 连接信号和槽

    def stop_thread(self):
        """
            关闭当前子进程
        """
        # 关闭程序时关闭出结束程序之外的其他按键
        self.pushButton_start_prog.setEnabled(True)
        self.pushButton_stop_prog.setEnabled(False)
        self.pushButton_start_test.setEnabled(True)
        self.pushButton_clear.setEnabled(True)
        if self.thread and self.thread.isRunning():
            self.check_distance = None
            # 发送停止信号给线程
            self.thread.stop()
            # 等待线程结束
            self.thread.wait()

    def startTest(self):
        """
            图片偏好性测试
        """
        self.img_select.select_hobby = True  # 使用了图片偏好性测试窗口
        try:
            self.sub_window_hobby_inst = subWindowHobby(
                self.img_select)  # 实例化窗口并调用
            if not self.sub_window_hobby_inst or not self.sub_window_hobby_inst.isVisible():
                self.sub_window_hobby_inst.show()
                self.sub_window_hobby_inst.setModal(True)  # 设置为模态窗口
        except:
            QMessageBox.critical(self, "错误", "图片偏好性测试过程出错!")

    def clearResult(self):
        """
            清除图片的偏好性
        """
        # print(self.img_select.subdirs_weight)
        self.img_select.clear_weight()

    def changeDistance(self, distance):
        distanceLimitation = 35                                 # 距离阈值
        # print(distance)
        if distance < distanceLimitation:
            if not self.sub_window_instance or not self.sub_window_instance.isVisible():
                self.sub_window_instance = subWindow(
                    self.img_select)                        # 实例化窗口并调用
                self.sub_window_instance.show()
                self.sub_window_instance.setModal(True)     # 设置为模态窗口
            else:
                pass
        elif distance >= distanceLimitation and self.sub_window_instance and self.sub_window_instance.isVisible():
            self.sub_window_instance.update_custom_hobby()
            self.sub_window_instance.close()                # 关闭窗口
            self.sub_window_instance = None

    def closeEvent(self, event):
        sys.exit(app.exec())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    imgs_folder = "software//imgs_folder"
    mainWin = mainWindow(imgs_folder)
    mainWin.show()
    sys.exit(app.exec())
