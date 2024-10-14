import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction, QDialog
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtGui import QPixmap, QImageReader
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from numpy import random

# from UI.detect_UI013 import Ui_MainWindow
# 分级
from UI.Ui_main_grade import Ui_MainWindow
# 控制
# from UI.Ui_main_control import Ui_MainWindow

from dialog.rtsp_win import Window
from models.experimental import attempt_load
from utils.CustomMessageBox import MessageBox
from utils.capnums import Camera
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from skimage.color import rgb2hsv, rgb2gray
from skimage.morphology import binary_opening, binary_closing, disk
from skimage.feature import canny

# 导入sklearn库中的MinMaxScaler函数，用于对数据进行归一化处理
from sklearn.preprocessing import MinMaxScaler


#这段代码定义了一个名为DetThread的类，并且使用pyqtSignal定义了几个信号，用于在目标检测过程中发送不同类型的数据到用户界面
class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    # 定义一个信号，用于发送统计信息
    send_statistic = pyqtSignal(dict)
    # 定义一个信号，用于发送分级统计结果!
    send_grade_statistic = pyqtSignal(dict)
    # 发送信号：正在检测/暂停/停止/检测结束/错误报告
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    #这个__init__方法是类的构造函数，用于初始化类的属性。它设置了各种属性，例如权重文件路径(weights)、
    # 当前权重文件路径(current_weight)、视频源(source)、置信度阈值(conf_thres)、IoU阈值(iou_thres)等等
    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = 'UI/best.pt'           # 设置权重
        self.current_weight = 'UI/best.pt'    # 当前权重
        self.source = '../inference/images'     # 视频源
        self.conf_thres = 0.45                  # 置信度
        self.iou_thres = 0.55                   # iou
        self.jump_out = False                   # 跳出循环
        self.is_continue = True                 # 继续/暂停
        self.percent_length = 1000              # 进度条
        self.rate_check = True                  # 是否启用延时
        self.rate = 100                         # 延时HZ
        self.guojing_thres = 0.4                # 果径权重
        self.yanse_thres = 0.3                  # 颜色权重
        self.yuanxingdu_thres = 0.3             # 圆形度权重


        # self.img_size = 640
        # self.device = 'cpu'
        # self.view_img = False
        # self.save_txt = False
        # self.save_conf = False
        # self.nosave = False
        # self.classes = None
        # self.agnostic_nms = False
        # self.augment = False
        # self.update = False
        # self.project = 'runs/detect'
        # self.name = 'exp'
        # self.exist_ok = True
        # self.no_trace = False
        # self.max_det=1000
        # self.hide_labels=False
        # self.hide_conf = False
        # self.line_thickness=1
        # self.save_crop=False
        # self.visualize=False
        # self.half=False





    @torch.no_grad()
    #这是一个run方法，它是在DetThread类中执行目标检测的主要方法。被QThread的start方法调用来启动线程
    def run(self,
            img_size = 640,
            device = 'cpu',
            view_img = False,
            save_txt = True,
            save_conf = False,
            nosave = False,
            classes = None,
            agnostic_nms = False,
            augment = False,
            update = False,
            project = 'runs/detect',
            name = 'exp',
            exist_ok = True,
            no_trace = False,
            max_det = 1000,
            hide_labels = False,
            hide_conf = False,
            line_thickness = 1,
            save_crop = False,
            visualize = False,
            half = False
    ):
        try:
            source, weights, view_img, save_txt, imgsz, trace, half = self.source, self.weights, view_img, save_txt, img_size, not no_trace, half
            save_img = not nosave and not source.endswith('.txt')  # save inference images
            webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))


            # Directories
            save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Initialize
            set_logging()
            device = select_device(device)
            half = device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(weights, map_location=device)  # load FP32 model

            num_params = 0
            for param in model.parameters():
                num_params += param.numel()

            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size

            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            if trace:
                model = TracedModel(model, device, img_size)

            if half:
                model.half()  # to FP16

            # Second-stage classifier


            # Set Dataloader
            vid_path, vid_writer = None, None
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                # dataset = LoadStreams(source, img_size=imgsz, stride=stride)
                dataset = LoadWebcam(source, img_size=imgsz, stride=stride)   #这里的视频流用什么加载，官方是上面注释的，案例是下面的这里暂且用了下面的
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride)

            # Get names and colors
            #names = model.module.names if hasattr(model, 'module') else model.names
            # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            old_img_w = old_img_h = imgsz
            old_img_b = 1

            count = 0
            # 跳帧检测
            jump_count = 0
            dataset = iter(dataset)

            t0 = time.time()

            while True:
                # 手动停止
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('停止')
                    break
                    # 临时更换模型
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(weights, map_location=device)  # load FP32 model

                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()

                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check img_size

                    names = model.module.names if hasattr(model, 'module') else model.names

                    if trace:
                        model = TracedModel(model, device, img_size)

                    if half:
                        model.half()  # to FP16



                    # Get names and colors
                    # names = model.module.names if hasattr(model, 'module') else model.names

                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                    self.current_weight = self.weights

                # 暂停开关
                # 这是主要的目标检测循环部分。在每次循环中，检查是否需要停止检测，检查当前权重文件是否发生了变化，检查是否可以继续检测。
                # 然后，获取下一帧图像，计算帧率，计算进度百分比，并将图像转换为PyTorch张量。接下来，将图像输入模型进行推断，应用非最大抑制，
                # 处理检测结果，并将结果发送到UI。在每次循环结束时，检查是否达到了检测结束的条件，如果是，则发送相应的信号。
                if self.is_continue:
                    path, img, im0s, self.vid_cap = next(dataset)
                    # for path, img, im0s, self.vid_cap in dataset:

                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])

                    # jump_count += 1
                    # if jump_count % 5 != 0:
                    #     continue
                    count += 1
                    # 每三十帧刷新一次输出帧率
                    if count % 30 == 0 and count >= 30:
                        fps = int(30 / (time.time() - t0))
                        self.send_fps.emit('fps：' + str(fps))
                        t0 = time.time()
                    if self.vid_cap:
                        percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    # 创建一个字典，键为name，值为0
                    statistic_dic = {name: 0 for name in names}
                    # 将img从numpy数组转换为张量，并将其转移到device上
                    img = torch.from_numpy(img).to(device)
                    # 如果half为真，则将img转换为半精度，否则转换为全精度
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    # 将img的值缩放到0.0 - 1.0的范围
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    # 如果img的维度为3，则将其调整为[1, 3, 128, 128]
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    #pred = model(img, augment=self.augment)[0]

                    # Warmup
                    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                        old_img_b = img.shape[0]
                        old_img_h = img.shape[2]
                        old_img_w = img.shape[3]
                        for i in range(3):
                            model(img, augment=augment)[0]

                    # Inference
                    t1 = time_synchronized()
                    # with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                    #     pred = model(img, augment=self.augment)[0]
                    pred = model(img, augment=augment)[0]
                    t2 = time_synchronized()

                    # pred = model(img, augment=self.augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms)

                    t3 = time_synchronized()


                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        im0 = im0s.copy()

                        # if webcam:  # batch_size >= 1
                        #     p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                        # else:
                        #     p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                        p, s, frame = path, '', getattr(dataset, 'frame', 0)

                        p = Path(p)  # to Path
                        save_path = str(save_dir / p.name)  # img.jpg
                        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                        -1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                if save_img or view_img:  # Add bbox to image
                                    label = f'{names[int(cls)]} {conf:.2f}'
                                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                                c = int(cls)  # integer class
                                statistic_dic[names[c]] += 1
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                # im0 = plot_one_box_PIL(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)  # 中文标签画框，但是耗时会增加
                                #plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=self.line_thickness)
                                plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=line_thickness)



                        # Print time (inference + NMS)
                        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')


                        # Save results (image with detections)
                        if save_img:
                            if dataset.mode == 'image':
                                cv2.imwrite(save_path, im0)
                                print(f" The image with the result is saved in: {save_path}")
                            else:  # 'video' or 'stream'
                                if vid_path != save_path:  # new video
                                    vid_path = save_path
                                    if isinstance(vid_writer, cv2.VideoWriter):
                                        vid_writer.release()  # release previous video writer
                                    if self.vid_cap:  # video
                                        fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
                                        w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    else:  # stream
                                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                                        save_path += '.mp4'
                                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                                vid_writer.write(im0)

                    if save_txt or save_img:
                        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                        # print(f"Results saved to {save_dir}{s}")

                    print(f'Done. ({time.time() - t0:.3f}s)')

                    # 控制视频发送频率
                    if self.rate_check:
                        time.sleep(1 / self.rate)
                    # print(type(im0s))
                    self.send_img.emit(im0)
                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    # 发送统计字典到主窗口
                    self.send_statistic.emit(statistic_dic)
                    if percent == self.percent_length:
                        self.send_percent.emit(0)
                        self.send_msg.emit('检测结束')
                        # 正常跳出循环
                        break
        except Exception as e:
            self.send_msg.emit('%s' % e)



#这里定义了一个名为MainWindow的类，继承自QMainWindow和Ui_MainWindow。这个类表示主窗口，并将UI布局加载到窗口中。
# __init__方法是类的构造函数，用于初始化窗口。
class MainWindow(QMainWindow, Ui_MainWindow, QDialog):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False
        # win10的CustomizeWindowHint模式，边框上面有一段空白。
        # 不想看到空白可以用FramelessWindowHint模式，但是需要重写鼠标事件才能通过鼠标拉伸窗口，比较麻烦
        # 不嫌麻烦可以试试, 写了一半不想写了，累死人
        self.setWindowFlags(Qt.CustomizeWindowHint)
        # self.setWindowFlags(Qt.FramelessWindowHint)
        # 自定义标题栏按钮
        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        self.closeButton.clicked.connect(self.close)

        # 定时清空自定义状态栏上的文字
        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # 自动搜索模型
        self.comboBox.clear()
        self.pt_list = os.listdir('D:/YOLO/yolov7-main-zyy/UI')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('D:/YOLO/yolov7-main-zyy/UI/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)


        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "D:/YOLO/yolov7-main-zyy/UI/%s" % self.model_type           # 权重
        self.det_thread.source = '0'                                    # 默认打开本机摄像头，无需保存到配置文件
        #self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        #self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.file.clicked.connect(self.open_file)
        self.camera.clicked.connect(self.chose_cam)
        #self.rtsp.clicked.connect(self.chose_rtsp)

        #self.project.clicked.connect(self.select_savepath)

        self.BeginDet.clicked.connect(self.run_or_continue)
        # self.BeginDet.clicked.connect(self.grading)
        self.StopDet.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)
        # self.comboBox.currentTextChanged.connect(lambda x: self.statistic_msg('模型切换为%s' % x))
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.guojingSlider.valueChanged.connect(lambda x: self.change_val(x, 'guojingSlider'))
        self.guojingSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'guojingSpinBox'))
        self.yanseSlider.valueChanged.connect(lambda x: self.change_val(x, 'yanseSlider'))
        self.yanseSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'yanseSpinBox'))
        self.yuanxingduSlider.valueChanged.connect(lambda x: self.change_val(x, 'yuanxingduSlider'))
        self.yuanxingduSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'yuanxingduSpinBox'))



        # 我的添加
        # self.file.clicked.connect(self.Selectfile)


        self.checkBox.clicked.connect(self.checkrate)
        self.load_setting()

    #用于搜索指定目录下的.pt文件，并将文件名添加到下拉列表框cbModels中供用户选择模型文件
    def search_pt(self):
        # pt_list = os.listdir('./pt')
        # pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list = os.listdir('D:/YOLO/yolov7-main-zyy/UI')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('D:/YOLO/yolov7-main-zyy/UI/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def checkrate(self):
        if self.checkBox.isChecked():
            # 选中时
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    #chose_rtsp方法用于选择RTSP视频流作为输入源。它通过QInputDialog.getText方法弹出对话框，要求用户输入RTSP地址，
    # 并将输入的地址存储在rtsp变量中。如果用户点击了对话框的确定按钮，ok值将为True，然后将device属性设置为输入的RTSP地址
    def chose_rtsp(self):
        self.rtsp_window = Window()
        # config_file = 'config/ip.json'
        config_file = 'D:/YOLO/yolov7-main-zyy/config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.67:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtsp.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='提示', text='请稍等，正在加载rtsp视频流', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('加载rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)

    #chose_cam方法用于选择本机摄像头作为输入源。它将device属性设置为'cam'，表示选择了摄像头
    def chose_cam(self):
        try:
            self.stop()
            # MessageBox的作用：留出2秒，让上一次摄像头安全release
            MessageBox(
                self.closeButton, title='提示', text='请稍等，正在检测摄像头设备', time=2000, auto=True).exec_()
            # 自动检测本机有哪些摄像头
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.camera.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.groupBox_5.mapToGlobal(self.camera.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.camera.pos()).y()
            y = y + self.camera.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('加载摄像头：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)

    # 导入配置文件
    #load_setting方法用于加载配置文件并设置界面上的参数值。首先，通过QFileDialog.getOpenFileName方法选择配置文件，
    # 并将结果存储在file_path中。然后，使用configparser.ConfigParser读取配置文件内容。如果配置文件中存在YOLOv5节（section），
    # 则获取该节下的各个参数值，并使用setCurrentText、setText方法设置界面上的对应控件的值。
    def load_setting(self):
        #config_file = 'config/setting.json'
        config_file = 'D:/YOLO/yolov7-main-zyy/config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            new_config = {"iou": 0.26,
                          "conf": 0.33,
                          "rate": 10,
                          "check": 0
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            iou = config['iou']
            conf = config['conf']
            rate = config['rate']
            check = config['check']
        self.confSpinBox.setValue(iou)
        self.iouSpinBox.setValue(conf)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check

    #change_val方法用于开始或暂停目标检测循环，并根据状态更新按钮的文本。如果循环正在运行（flag为True），则将其暂停，并将按钮文本设置为'开始'。
    # 如果循环未运行（flag为False），则将其开始，并将按钮文本设置为'暂停'。在设置参数值之前，使用mutex.lock方法获取互斥锁，确保线程安全。
    # 然后，使用setVal方法设置目标检测模型的参数值。最后，使用mutex.unlock方法释放互斥锁。
    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x*100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x/100)
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x*100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x/100)
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
        elif flag == 'guojingSpinBox':
            self.guojingSlider.setValue(int(x*100))
        elif flag == 'guojingSlider':
            self.guojingSpinBox.setValue(x/100)
        elif flag == 'yanseSpinBox':
            self.yanseSlider.setValue(int(x*100))
        elif flag == 'yanseSlider':
            self.yanseSpinBox.setValue(x/100)
        elif flag == 'checkBox':
            self.det_thread.rate_check = x
        elif flag == 'yuanxingduSpinBox':
            self.yuanxingduSlider.setValue(int(x*100))
        elif flag == 'yuanxingduSlider':
            self.yuanxingduSpinBox.setValue(x/100)

        # elif flag == 'x_SpinBox':
        #     self.x_Slider.setValue(int(x*100))
        # elif flag == 'x_Slider':
        #     self.x_SpinBox.setValue(x/100)
        # elif flag == 'y_SpinBox':
        #     self.y_Slider.setValue(int(x*100))
        # elif flag == 'y_Slider':
        #     self.y_SpinBox.setValue(x/100)
        # elif flag == 'z_SpinBox':
        #     self.z_Slider.setValue(int(x*100))
        # elif flag == 'z_Slider':
        #     self.z_SpinBox.setValue(x/100)
        elif flag == 'x_SpinBox':
            x = max(min(x, 1), -1)
            self.x_Slider.setValue(int(x * 100))
        elif flag == 'x_Slider':
            x = max(min(x / 100, 1), -1)
            self.x_SpinBox.setValue(x)
        elif flag == 'y_SpinBox':
            y = max(min(y, 1), -1)
            self.y_Slider.setValue(int(y * 100))
        elif flag == 'y_Slider':
            y = max(min(y / 100, 1), -1)
            self.y_SpinBox.setValue(y)
        elif flag == 'z_SpinBox':
            z = max(min(z, 1), 0)
            self.z_Slider.setValue(int(z * 100))
        elif flag == 'z_Slider':
            z = max(min(z / 100, 1), 0)
            self.z_SpinBox.setValue(z)


        
        


        # 这段代码是用于更新三个SpinBox（数值框）的值的。首先，它会计算三个SpinBox的值的总和，然后检查总和是否等于1。如果不等于1，它会根据给定的标志（flag）来调整其他两个SpinBox的值，以使总和等于1。
        # 具体来说，代码首先计算总和，然后检查总和是否等于1。如果不等于1，它会计算差值（1减去总和），然后根据标志来调整其他两个SpinBox的值。如果标志是'guojingSpinBox'，那么它会将'yanseSpinBox'的值增加差值的一半，并将'yuanxingduSpinBox'的值增加差值的一半。如果标志是'yanseSpinBox'，那么它会将'guojingSpinBox'的值增加差值的一半，并将'yuanxingduSpinBox'的值增加差值的一半。如果标志是'yuanxingduSpinBox'，那么它会将'guojingSpinBox'的值增加差值的一半，并将'yanseSpinBox'的值增加差值的一半。
        # 总之，这段代码的作用是确保三个SpinBox的值之和等于1，同时保持它们的相对大小关系不变。
        # 更新总和
        self.sum_values = self.guojingSpinBox.value() + self.yanseSpinBox.value() + self.yuanxingduSpinBox.value()

        # 如果总和不等于1，调整其他两个值
        if self.sum_values != 1:
            diff = 1 - self.sum_values
            if flag == 'guojingSpinBox':
                self.yanseSpinBox.setValue(self.yanseSpinBox.value() + diff/2)
                self.yuanxingduSpinBox.setValue(self.yuanxingduSpinBox.value() + diff/2)
            elif flag == 'yanseSpinBox':
                self.guojingSpinBox.setValue(self.guojingSpinBox.value() + diff/2)
                self.yuanxingduSpinBox.setValue(self.yuanxingduSpinBox.value() + diff/2)
            elif flag == 'yuanxingduSpinBox':
                self.guojingSpinBox.setValue(self.guojingSpinBox.value() + diff/2)
                self.yanseSpinBox.setValue(self.yanseSpinBox.value() + diff/2)



    # 选择照片/视频 并展示，这是D:\YOLO\yolov7-Pyside6-main-grade\recognizition.py文件中的函数。
    # def Selectfile(self):
    #     file, _ = QFileDialog.getOpenFileName(
    #         self,  # 父窗口对象
    #         "选择你要上传的图片/视频",  # 标题
    #         "./",  # 默认打开路径为当前路径
    #         "图片/视频类型 (*.jpg *.jpeg *.png *.bmp *.dib  *.jpe  *.jp2 *.mp4)"  # 选择类型过滤项，过滤内容在括号中
    #     )
    #     if file == "":
    #         pass
    #     else:
    #         self.inputPath = file
    #         glo.set_value('inputPath', self.inputPath)
    #         if ".avi" in self.inputPath or ".mp4" in self.inputPath:
    #             # 显示第一帧
    #             self.cap = cv2.VideoCapture(self.inputPath)
    #             ret, frame = self.cap.read()
    #             if ret:
    #                 rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                 self.showimg(rgbImage, self.input, 'img')
    #         else:
    #             self.showimg(self.inputPath, self.input, 'path')




    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)   # 3秒后自动清除

    def show_msg(self, msg):
        self.BeginDet.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "D:/YOLO/yolov7-main-zyy/UI/%s" % self.model_type
        self.statistic_msg('模型切换为%s' % x)

    def select_savepath(self):
        folder = QFileDialog.getExistingDirectory(self, '选择路径', '.')
        self.save_path = folder
        self.save_id = 0
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def open_file(self, name):
        ### 每次打开文件前，先清空labels文件夹和crop文件夹
        labels_folder = 'runs/detect/exp/labels'
        for file_name in os.listdir(labels_folder):
            file_path = os.path.join(labels_folder, file_name)
            os.remove(file_path)

        output_dir = 'mnt/data/crop'  # 请替换为你希望保存的目录
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            os.remove(file_path)

        ### 每次打开文件前，清空两个结果框
        self.resultWidget.clear()
        self.gradeWidget.clear()
        

        # source = QFileDialog.getOpenFileName(self, '选取视频或图片', os.getcwd(), "Pic File(*.mp4 *.mkv *.avi *.flv "
        #                                                                    "*.jpg *.png)")
        config_file = 'D:/YOLO/yolov7-main-zyy/config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        # 判断文件夹是否存在, 如果不存在则使用当前路径
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        # 获取用户选择的文件路径
        name, _ = QFileDialog.getOpenFileName(self, '选取视频或图片', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")
        # 判断用户是否选择了文件, 如果没有选择则不执行任何操作, 直接返回
        if name:
            self.det_thread.source = name
            self.statistic_msg('加载完成：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            # 切换文件后，上一次检测停止
            self.stop()

        ## 自己添加的。
        ##模板：self.showimg(self.inputPath, self.input, 'path')
            ## self.showimg(name, self.raw_video, 'path')

        # ## 用于显示图片，预览
        # if '.jpg' in name or '.png' in name:
        #     # 创建 QGraphicsView
        #     self.raw_video = QGraphicsView(self)
        #     self.raw_video.setRenderHint(QPainter.Antialiasing)
        #     self.raw_video.setRenderHint(QPainter.SmoothPixmapTransform)
        #     self.raw_video.setRenderHint(QPainter.TextAntialiasing)

        #     # 创建 QGraphicsPixmapItem
        #     pixmap = QPixmap(name)
        #     item = pg.QtGui.QGraphicsPixmapItem(pixmap)

        #     # 将 QGraphicsPixmapItem 添加到 QGraphicsView
        #     self.raw_video.scene().addItem(item)

        #     # 显示 QGraphicsView
        #     self.raw_video.show()

            #
            # 最大化界面与正常化界面函数，下面这个函数是原本别人的，但是实际应用中这个界面实现不了，所以在后面又重新找了一个正常可用的函数
            #     def max_or_restore(self):
            #         if self.maxButton.isChecked():
            #             self.showMaximized()
            #         else:
            #             self.showNormal()


    # 显示Label图片
    #@staticmethod
    def showimg(img, label, flag):
        try:
            if flag == "path":
                img_src = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
            else:
                img_src = img
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))    



    def max_or_restore(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    # 继续/暂停
    def run_or_continue(self):
        # 设置线程的跳出标志为False
        self.det_thread.jump_out = False
        # 如果按钮被点击
        if self.BeginDet.isChecked():
            # 设置线程的继续标志为True
            self.det_thread.is_continue = True
            # 如果线程没有运行
            if not self.det_thread.isRunning():
                # 启动线程
                self.det_thread.start()
            # 获取源文件名
            source = os.path.basename(self.det_thread.source)
            # 如果源文件名是数字，则将其显示为摄像头设备，否则显示为源文件名
            source = '摄像头设备' if source.isnumeric() else source
            # 显示正在检测的模型和文件
            self.statistic_msg('正在检测 >> 模型：{}，文件：{}'.format(os.path.basename(self.det_thread.weights),source))
        else:
            # 设置线程的继续标志为False
            self.det_thread.is_continue = False
            # 显示暂停
            self.statistic_msg('暂停')

    # 退出检测循环
    def stop(self):
        self.det_thread.jump_out = True

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            # QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        # self.setCursor(QCursor(Qt.ArrowCursor))

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # 保持纵横比
            # 找出长边
            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))


        ###########################
    def true_grading(name):
        # 1. 读取本地文件，展示RGB模型下的苹果图
        # 请替换为你的图像路径，这里的name就是连接‘选择文件’和‘获取文件路径’的连接（锁链）。
        image_path = name  

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # show_image('RGB Image', image_rgb)

        # 2. 展示HSI模型下的苹果图
        image_hsv = rgb2hsv(image_rgb)
        # show_image('HSI Image', image_hsv)

        # 3. 展示苹果灰度图像
        image_gray = rgb2gray(image_rgb)
        # show_image('Grayscale Image', image_gray, cmap='gray')

        # 4. 采用双边滤波对图像进行去噪
        image_filtered = cv2.bilateralFilter(image_gray.astype(np.float32), 9, 75, 75)
        # show_image('Bilateral Filtered Image', image_filtered, cmap='gray')


        # 5. 采用颜色分割将苹果和背景分割开
        h_channel = image_hsv[:, :, 0]
        red_mask = ((h_channel >= 0.0) & (h_channel <= 0.172)) | ((h_channel >= 0.8) & (h_channel <= 1.0))
        binary = red_mask
        # show_image('Color Segmentation', binary, cmap='gray')


        # 6. 采用形态学操作膨胀和腐蚀去除噪声，平滑二值图像边缘
        selem = disk(7)
        binary_opened = binary_opening(binary, selem)
        binary_closed = binary_closing(binary_opened, selem)
        # show_image('Morphological Operations', binary_closed, cmap='gray')


        # 7. 采用Canny边缘检测算法
        edges = canny(binary_closed)
        # show_image('Canny Edges', edges, cmap='gray')

        # 查找轮廓并绘制最小外接圆
        edges_uint8 = (edges * 255).astype(np.uint8)
        contours, _ = cv2.findContours(edges_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 在边缘检测图像上绘制最小外接圆
        output_image = cv2.cvtColor(edges_uint8, cv2.COLOR_GRAY2RGB)  # 将边缘检测图像转换为RGB格式
        for contour in contours:
            if len(contour) > 0:
                center, radius = cv2.minEnclosingCircle(contour)
                center = (int(center[0]), int(center[1]))
                radius = int(radius)
                cv2.circle(output_image, center, radius, (255, 0, 0), 2)  # 以红色绘制最小外接圆


        red_mask = ((h_channel >= 0.0) & (h_channel <= 0.06)) | ((h_channel >= 0.97) & (h_channel <= 1.0))
        color_ratio = np.sum(red_mask) / red_mask.size
        #print(f"苹果的着色率: {color_ratio}")

        # 10. 圆形度E的计算。标准圆的周长和面积对应的半径是相同的，圆形状越是规则，两个半径相差就越小，因此，可以利用同一区域的面积 和 周 长 各 自 对 应 半 径 的 比 值 来 计 算 圆 形 度。

        perimeter = np.sum(edges)
        area = np.sum(binary_closed)
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        diameter = perimeter / (np.pi)
        #print(f"苹果的半径: {diameter}")  
        #print(f"苹果的圆形度: {circularity}")

        diameter = round(diameter, 2)
        color_ratio = round(color_ratio, 2)
        circularity = round(circularity, 2)

        return diameter, color_ratio, circularity


        ###########################


    # 要输入一个文件路径，里面是直接经行计算的到参数的图片，就是裁剪过的图片。
    # 所以folder_path = 裁剪输出的文件夹。
    def process_images_in_cropped_folder(folder_path):
        apple_parameters = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            image = cv2.imread(file_path)
            if image is not None:
                apple_parameters.append(MainWindow.true_grading(file_path))
            else:
                print(f"无法读取图像，请检查路径：{file_path}")

        return apple_parameters

        ###########################

    def grading(self):
        # #########
        # ### 弹窗，选择文件。
        # config_file = 'D:/YOLO/yolov7-main-zyy/config/fold.json'
        # # config = json.load(open(config_file, 'r', encoding='utf-8'))
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        # open_fold = config['open_fold']
        # # 判断文件夹是否存在, 如果不存在则使用当前路径
        # if not os.path.exists(open_fold):
        #     open_fold = os.getcwd()
        # # 获取用户选择的文件路径，QFileDialog.getOpenFileName()函数会弹出一个文件选择对话框，让用户选择一个文件。
        # # 如果用户选择了一个文件，那么name变量就会保存用户选择的文件的路径。
        # name, _ = QFileDialog.getOpenFileName(self, '选取视频或图片', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
        #                                                                   "*.jpg *.png)")
        # # 判断用户是否选择了文件, 如果没有选择则不执行任何操作, 直接返回
        # if name:
        #     self.det_thread.source = name
        #     self.statistic_msg('加载完成：{}'.format(os.path.basename(name)))
        #     config['open_fold'] = os.path.dirname(name)
        #     config_json = json.dumps(config, ensure_ascii=False, indent=2)
        #     with open(config_file, 'w', encoding='utf-8') as f:
        #         f.write(config_json)
        #     # 切换文件后，上一次检测停止
        #     self.stop()        
        # #########

        # 获取源文件名
        # name = os.path.basename(self.det_thread.source)
        name = self.det_thread.source 
        

        #########
        ### 图像裁剪
        # 加载包含检测框的图像
        # image_path = 'runs/detect/exp16/067.jpg'  # 请替换为你的图像路径
        image_path = name
        image = cv2.imread(image_path)

        if image is not None:
            image_height, image_width = image.shape[:2]
            
            # labels_path = 'runs/detect/exp16/labels/067.txt'  # 请替换为你的标签文件路径
            # 获取文件夹中的文件名
            labels_folder = 'runs/detect/exp/labels'
            
            # 获取指定文件夹中的所有文件名
            file_name = os.listdir(labels_folder)[0]

            # 获取文件的完整路径
            labels_path = os.path.join(labels_folder, file_name)

            # 读取标签文件
            boxes = MainWindow.read_labels(labels_path, image_width, image_height)      

            # 裁剪检测框内的图像并保存
            output_dir = 'mnt/data/crop'  # 请替换为你希望保存的目录
            # for file_name in os.listdir(output_dir):
            #     file_path = os.path.join(output_dir, file_name)
            #     os.remove(file_path)

            MainWindow.crop_and_save(image, boxes, output_dir)
            
            # 显示原始图像
            # show_image('Original Image', image)
        else:
            print(f"无法读取图像，请检查路径：{image_path}")  
        # 以上的函数经行完，就已经生成完成了裁剪后的图片，并且保存在了output_dir = 'mnt/data/crop'中。
        #########
        # 遍历裁剪图片文件夹，并执行分级，返回分级的三个参数
        cropped_image_folder_path = 'mnt/data/crop'
        folder_path = cropped_image_folder_path
        apple_parameters = MainWindow.process_images_in_cropped_folder(folder_path)
        

        return apple_parameters
  


    def read_labels(file_path, image_width, image_height):
        boxes = []
        with open(file_path, 'r') as file:
            for line in file:
                data = line.strip().split()
                label = int(data[0])
                x_center = float(data[1]) * image_width
                y_center = float(data[2]) * image_height
                width = float(data[3]) * image_width
                height = float(data[4]) * image_height
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                boxes.append((label, x1, y1, x2, y2))
        return boxes

    def crop_and_save(image, boxes, output_dir):
        for i, (label, x1, y1, x2, y2) in enumerate(boxes):
            cropped_image = image[y1:y2, x1:x2]
            output_path = f"{output_dir}/cropped_image_{i}.png"
            cv2.imwrite(output_path, cropped_image)
            # show_image(f'Cropped Image {i}', cropped_image)


        



    

    def apple_grading(self, diameter, color_ratio, circularity):
        # 假设这个方法根据三个参数返回一个分级结果
        # self.guojingSpinBox.value() , self.yanseSpinBox.value() , self.yuanxingduSpinBox.value()
        value_a = self.guojingSpinBox.value()
        value_b = self.yanseSpinBox.value()
        value_c = self.yuanxingduSpinBox.value()
        print("value_a:", value_a)
        print("value_b:", value_b)
        print("value_c:", value_c)

        

        if diameter >= 400 * value_a and color_ratio >= 0.8 * value_b and circularity >= 0.8 * value_c:
            return "特级果"
        elif diameter >= 300 * value_a and color_ratio >= 0.6 * value_b and circularity >= 0.6 * value_c:
            return "一级果"
        else:
            return "二级果"
    

    # 实时统计
    def show_statistic(self, statistic_dic):
        '''显示统计信息
        参数:statistic_dic:统计字典
        '''
        try:

            #### 第一部分：结果统计框
            # 先清空labels文件夹中的所有文件 
            # labels_folder = 'runs/detect/exp/labels'
            # for file_name in os.listdir(labels_folder):
            #     file_path = os.path.join(labels_folder, file_name)
            #     os.remove(file_path)


            self.resultWidget.clear()
            self.gradeWidget.clear()
            # 按值从高到低排序
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            # 过滤掉值为0的项
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            # 创建显示结果列表
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            # 显示结果,加载到resultWidget框中。
            self.resultWidget.addItems(results)


            #### 第二部分：后台经行分级
            ### 分级展示：
                
            # ### 图像裁剪
            #在MainWindow.grading()函数中

            # 获取苹果参数，手动选择文件，经行图像预处理以及评分和分级。
            # apple_parameters = MainWindow.grading(self)


            # 处理裁剪后的图像文件夹中的所有图像
            cropped_images_folder = 'mnt/data/crop'
            

            apple_parameters = MainWindow.grading(self)
            # apple_parameters = DetThread.grading()
            # 对每个苹果进行分级（这里要改，有问题）
            # 将apple_parameters中的每个元素与self.apple_grading(*apple)的结果组合成一个新的元组，并将新的元组添加到graded_apples列表中
            graded_apples = [(i+1,) + apple + (self.apple_grading(*apple),) for i, apple in enumerate(apple_parameters)]

            

            #### 第三部分：生成评分表格
            # 输出分级结果
            print("序号\t果径\t颜色\t圆形度\t分级结果")
            for apple in graded_apples:
                print(f"{apple[0]}\t{apple[1]}\t{apple[2]}\t{apple[3]}\t{apple[4]}")

            # 将分级结果添加到gradeWidget
            grading_results = ['序号\t果径\t颜色\t圆形度\t分级结果']
            for apple in graded_apples:
                grading_results.append(f'{apple[0]}\t{apple[1]}\t{apple[2]}\t{apple[3]}\t{apple[4]}')

            # 清空gradeWidget
            self.gradeWidget.clear()

            self.gradeWidget.addItems(grading_results)
           


            
        except Exception as e:
            print(repr(e))






    # 关于计算分级的函数
    # 分级：特征计算

    def closeEvent(self, event):
        # 如果摄像头开着，先把摄像头关了再退出，否则极大可能可能导致检测线程未退出
        self.det_thread.jump_out = True
        # 退出时，保存设置
        config_file = 'D:/YOLO/yolov7-main-zyy/config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            self.closeButton, title='提示', text='正在关闭程序......', time=1000, auto=True).exec_()
        sys.exit(0)



class FuzzySVM:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        # 初始化参数
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.model = None
        # 创建一个全局缩放器
        # 导入MinMaxScaler类
        self.scaler = MinMaxScaler()  


    def fit(self, X, y, fuzzy_membership):
        # 缩放数据到[0, 1]范围
        X_scaled = self.scaler.fit_transform(X)

        # 根据模糊隶属度调整C值
        C_values = self.C * fuzzy_membership

        # 创建并训练模型，使用模糊隶属度作为样本权重
        # self.model = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma)
        self.model.fit(X_scaled, y, sample_weight=fuzzy_membership)

    def predict(self, X):
        # 缩放数据到[0, 1]范围
        X_scaled = self.scaler.transform(X)

        # 使用训练好的模型进行预测
        return self.model.predict(X_scaled)

    def decision_function(self, X):
        # 缩放数据到[0, 1]范围
        X_scaled = self.scaler.transform(X)

        # 获取决策函数值
        return self.model.decision_function(X_scaled)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
