# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'detect_UI_2.3.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1072, 786)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setStyleSheet(" QLabel {\n"
"        border: 2px solid #4CAF50; /* 给 logo 添加绿色边框 */\n"
"        border-radius: 10px; /* 设置边框圆角 */\n"
"        padding: 5px; /* 设置内边距 */\n"
"    }")
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap(":/png/icon/上海大学002.png"))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.minButton = QtWidgets.QPushButton(self.centralwidget)
        self.minButton.setEnabled(True)
        self.minButton.setMinimumSize(QtCore.QSize(50, 28))
        self.minButton.setMaximumSize(QtCore.QSize(50, 28))
        self.minButton.setStyleSheet("QPushButton {\n"
"    border-style: solid;\n"
"    border-width: 0px;\n"
"    border-radius: 5px;\n"
"    background-color: #FFFFFF; /* 使用白色背景 */\n"
"    color: white; /* 使用白色文字，与白色背景形成对比 */\n"
"}\n"
"\n"
"QPushButton::focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton::hover {\n"
"    border-style: solid;\n"
"    border-width: 0px;\n"
"    border-radius: 5px;\n"
"    background-color: rgba(60, 60, 60, 255); /* 当鼠标悬停时，使按钮变得稍微深色，提供视觉反馈 */\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: rgba(100, 100, 100, 255); /* 按钮被按下时，使按钮变得稍微浅色，提供视觉反馈 */\n"
"}\n"
"")
        self.minButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/png/icon/最小化_黑色.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.minButton.setIcon(icon)
        self.minButton.setObjectName("minButton")
        self.horizontalLayout.addWidget(self.minButton)
        self.maxButton = QtWidgets.QPushButton(self.centralwidget)
        self.maxButton.setMinimumSize(QtCore.QSize(50, 28))
        self.maxButton.setMaximumSize(QtCore.QSize(50, 28))
        self.maxButton.setStyleSheet("QPushButton {\n"
"    border-style: solid;\n"
"    border-width: 0px;\n"
"    border-radius: 5px;\n"
"    background-color: #FFFFFF; /* 使用白色背景 */\n"
"    color: white; /* 使用白色文字，与白色背景形成对比 */\n"
"}\n"
"\n"
"QPushButton::focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton::hover {\n"
"    border-style: solid;\n"
"    border-width: 0px;\n"
"    border-radius: 5px;\n"
"    background-color: rgba(60, 60, 60, 255); /* 当鼠标悬停时，使按钮变得稍微深色，提供视觉反馈 */\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: rgba(100, 100, 100, 255); /* 按钮被按下时，使按钮变得稍微浅色，提供视觉反馈 */\n"
"}\n"
"")
        self.maxButton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/png/icon/正方形_黑色.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.maxButton.setIcon(icon1)
        self.maxButton.setObjectName("maxButton")
        self.horizontalLayout.addWidget(self.maxButton)
        self.closeButton = QtWidgets.QPushButton(self.centralwidget)
        self.closeButton.setMinimumSize(QtCore.QSize(50, 28))
        self.closeButton.setMaximumSize(QtCore.QSize(50, 28))
        self.closeButton.setStyleSheet("QPushButton {\n"
"    border-style: solid;\n"
"    border-width: 0px;\n"
"    border-radius: 5px;\n"
"    background-color: #FFFFFF; /* 使用白色背景 */\n"
"    color: white; /* 使用白色文字，与白色背景形成对比 */\n"
"}\n"
"\n"
"QPushButton::focus {\n"
"    outline: none;\n"
"}\n"
"\n"
"QPushButton::hover {\n"
"    border-style: solid;\n"
"    border-width: 0px;\n"
"    border-radius: 5px;\n"
"    background-color: rgba(60, 60, 60, 255); /* 当鼠标悬停时，使按钮变得稍微深色，提供视觉反馈 */\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: rgba(100, 100, 100, 255); /* 按钮被按下时，使按钮变得稍微浅色，提供视觉反馈 */\n"
"}\n"
"")
        self.closeButton.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/png/icon/关闭_黑色.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.closeButton.setIcon(icon2)
        self.closeButton.setObjectName("closeButton")
        self.horizontalLayout.addWidget(self.closeButton)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 1)
        self.horizontalLayout.setStretch(3, 1)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMinimumSize(QtCore.QSize(400, 50))
        font = QtGui.QFont()
        font.setFamily("AcadEref")
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("QLabel {\n"
"    color: #000000; /* 黑色文字 */\n"
"    font-size: 20px; /* 20像素字体大小 */\n"
"    font-weight: bold; /* 加粗字体 */\n"
"    text-align: center; /* 文字居中对齐 */\n"
"    qproperty-alignment: AlignCenter; /* 属性对齐方式：居中 */\n"
"    border: 1px solid #4CAF50; /* 绿色边框 */\n"
"    border-radius: 10px; /* 10像素圆角 */\n"
"    padding: 10px; /* 10像素内边距 */\n"
"    background-color: #DCEEFB; /* 淡蓝色背景 */\n"
"}\n"
"")
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setMinimumSize(QtCore.QSize(150, 100))
        self.groupBox_2.setMaximumSize(QtCore.QSize(16777215, 300))
        self.groupBox_2.setStyleSheet("QGroupBox {\n"
"    font-size: 18px; /* 设置字体大小 */\n"
"    font-weight: bold; /* 字体加粗 */\n"
"    border: 1px solid #4CAF50; /* 设置边框为绿色 */\n"
"    border-radius: 10px; /* 设置圆角为10px */\n"
"    padding: 10px; /* 设置内边距为10px */\n"
"    margin-top: 10px; /* 设置顶部外边距为10px */\n"
"    background-color: #D3D3D3; /* 背景颜色为淡灰色 */\n"
"}\n"
"")
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox.setEnabled(True)
        self.comboBox.setMinimumSize(QtCore.QSize(0, 30))
        self.comboBox.setMaximumSize(QtCore.QSize(16777215, 30))
        self.comboBox.setStyleSheet("")
        self.comboBox.setObjectName("comboBox")
        self.verticalLayout.addWidget(self.comboBox)
        self.file = QtWidgets.QPushButton(self.groupBox_2)
        self.file.setMinimumSize(QtCore.QSize(0, 35))
        self.file.setMaximumSize(QtCore.QSize(16777215, 35))
        self.file.setStyleSheet("")
        self.file.setIconSize(QtCore.QSize(20, 20))
        self.file.setObjectName("file")
        self.verticalLayout.addWidget(self.file)
        self.camera = QtWidgets.QPushButton(self.groupBox_2)
        self.camera.setMinimumSize(QtCore.QSize(90, 35))
        self.camera.setMaximumSize(QtCore.QSize(16777215, 35))
        self.camera.setStyleSheet("")
        self.camera.setObjectName("camera")
        self.verticalLayout.addWidget(self.camera)
        self.horizontalLayout_7.addWidget(self.groupBox_2)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMinimumSize(QtCore.QSize(150, 200))
        self.groupBox.setMaximumSize(QtCore.QSize(16777215, 300))
        self.groupBox.setStyleSheet("QGroupBox {\n"
"    font-size: 18px; /* 设置字体大小 */\n"
"    font-weight: bold; /* 字体加粗 */\n"
"    border: 1px solid #4CAF50; /* 设置边框为绿色 */\n"
"    border-radius: 10px; /* 设置圆角为10px */\n"
"    padding: 10px; /* 设置内边距为10px */\n"
"    margin-top: 10px; /* 设置顶部外边距为10px */\n"
"    background-color: #D3D3D3; /* 背景颜色为淡灰色 */\n"
"}\n"
"")
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setMinimumSize(QtCore.QSize(25, 20))
        self.label_2.setStyleSheet("QLabel {\n"
"    font-size: 14px; /* 字体大小为14px */\n"
"    color: #000000; /* 字体颜色为黑色 */\n"
"}\n"
"")
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 0, 0, 1, 1)
        self.iouSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.iouSpinBox.setMinimumSize(QtCore.QSize(0, 25))
        self.iouSpinBox.setMaximumSize(QtCore.QSize(50, 16777215))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(8)
        self.iouSpinBox.setFont(font)
        self.iouSpinBox.setStyleSheet("x")
        self.iouSpinBox.setMaximum(1.0)
        self.iouSpinBox.setSingleStep(0.01)
        self.iouSpinBox.setProperty("value", 0.45)
        self.iouSpinBox.setObjectName("iouSpinBox")
        self.gridLayout_3.addWidget(self.iouSpinBox, 1, 0, 1, 1)
        self.iouSlider = QtWidgets.QSlider(self.groupBox)
        self.iouSlider.setMinimumSize(QtCore.QSize(80, 20))
        self.iouSlider.setStyleSheet("QSlider {\n"
"    min-height: 20px;\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"    height: 5px;\n"
"    background: #d3d3d3;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    width: 18px;\n"
"    margin: -7px 0; /* handle is placed by default on the contents rect of the groove. Expand outside the groove */\n"
"    background: #727272;\n"
"    border-radius: 9px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #fff;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background: #727272;\n"
"}\n"
"")
        self.iouSlider.setMaximum(100)
        self.iouSlider.setProperty("value", 45)
        self.iouSlider.setOrientation(QtCore.Qt.Horizontal)
        self.iouSlider.setObjectName("iouSlider")
        self.gridLayout_3.addWidget(self.iouSlider, 1, 1, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setMinimumSize(QtCore.QSize(45, 20))
        self.label_3.setStyleSheet("QLabel {\n"
"    font-size: 14px; /* 字体大小为14px */\n"
"    color: #000000; /* 字体颜色为黑色 */\n"
"}\n"
"")
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 2, 0, 1, 1)
        self.confSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.confSpinBox.setMinimumSize(QtCore.QSize(0, 25))
        self.confSpinBox.setMaximumSize(QtCore.QSize(50, 16777215))
        font = QtGui.QFont()
        font.setFamily("Microsoft JhengHei UI")
        font.setPointSize(8)
        self.confSpinBox.setFont(font)
        self.confSpinBox.setStyleSheet("")
        self.confSpinBox.setMaximum(1.0)
        self.confSpinBox.setSingleStep(0.01)
        self.confSpinBox.setProperty("value", 0.25)
        self.confSpinBox.setObjectName("confSpinBox")
        self.gridLayout_3.addWidget(self.confSpinBox, 3, 0, 1, 1)
        self.confSlider = QtWidgets.QSlider(self.groupBox)
        self.confSlider.setMinimumSize(QtCore.QSize(80, 20))
        self.confSlider.setStyleSheet("QSlider {\n"
"    min-height: 20px;\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"    height: 5px;\n"
"    background: #d3d3d3;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    width: 18px;\n"
"    margin: -7px 0; /* handle is placed by default on the contents rect of the groove. Expand outside the groove */\n"
"    background: #727272;\n"
"    border-radius: 9px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #fff;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background: #727272;\n"
"}\n"
"")
        self.confSlider.setMaximum(100)
        self.confSlider.setProperty("value", 25)
        self.confSlider.setOrientation(QtCore.Qt.Horizontal)
        self.confSlider.setObjectName("confSlider")
        self.gridLayout_3.addWidget(self.confSlider, 3, 1, 1, 2)
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setMinimumSize(QtCore.QSize(0, 20))
        self.label_5.setStyleSheet("QLabel {\n"
"    font-size: 14px; /* 字体大小为14px */\n"
"    color: #000000; /* 字体颜色为黑色 */\n"
"}\n"
"")
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 4, 0, 1, 2)
        self.checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox.setMinimumSize(QtCore.QSize(0, 25))
        self.checkBox.setStyleSheet("")
        self.checkBox.setChecked(True)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout_3.addWidget(self.checkBox, 4, 2, 1, 1)
        self.rateSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.rateSpinBox.setMinimumSize(QtCore.QSize(0, 25))
        self.rateSpinBox.setMaximumSize(QtCore.QSize(50, 16777215))
        font = QtGui.QFont()
        font.setFamily("Microsoft JhengHei UI")
        font.setPointSize(8)
        self.rateSpinBox.setFont(font)
        self.rateSpinBox.setStyleSheet("")
        self.rateSpinBox.setDecimals(2)
        self.rateSpinBox.setMinimum(1.0)
        self.rateSpinBox.setMaximum(20.0)
        self.rateSpinBox.setSingleStep(1.0)
        self.rateSpinBox.setObjectName("rateSpinBox")
        self.gridLayout_3.addWidget(self.rateSpinBox, 5, 0, 1, 1)
        self.rateSlider = QtWidgets.QSlider(self.groupBox)
        self.rateSlider.setMinimumSize(QtCore.QSize(80, 20))
        self.rateSlider.setStyleSheet("QSlider {\n"
"    min-height: 20px;\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"    height: 5px;\n"
"    background: #d3d3d3;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    width: 18px;\n"
"    margin: -7px 0; /* handle is placed by default on the contents rect of the groove. Expand outside the groove */\n"
"    background: #727272;\n"
"    border-radius: 9px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #fff;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background: #727272;\n"
"}\n"
"")
        self.rateSlider.setMaximum(20)
        self.rateSlider.setPageStep(1)
        self.rateSlider.setProperty("value", 1)
        self.rateSlider.setOrientation(QtCore.Qt.Horizontal)
        self.rateSlider.setObjectName("rateSlider")
        self.gridLayout_3.addWidget(self.rateSlider, 5, 1, 1, 2)
        self.horizontalLayout_7.addWidget(self.groupBox)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setMinimumSize(QtCore.QSize(150, 100))
        self.groupBox_3.setMaximumSize(QtCore.QSize(16777215, 300))
        self.groupBox_3.setStyleSheet("QGroupBox {\n"
"    font-size: 18px; /* 设置字体大小 */\n"
"    font-weight: bold; /* 字体加粗 */\n"
"    border: 1px solid #4CAF50; /* 设置边框为绿色 */\n"
"    border-radius: 10px; /* 设置圆角为10px */\n"
"    padding: 10px; /* 设置内边距为10px */\n"
"    margin-top: 10px; /* 设置顶部外边距为10px */\n"
"    background-color: #D3D3D3; /* 背景颜色为淡灰色 */\n"
"}\n"
"")
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.StopDet = QtWidgets.QPushButton(self.groupBox_3)
        self.StopDet.setMinimumSize(QtCore.QSize(90, 25))
        self.StopDet.setStyleSheet("QPushButton#BeginDet {\n"
"    background-color: #4CAF50;  /* 设置背景颜色为绿色 */\n"
"    border: none;  /* 无边框 */\n"
"    color: white;  /* 文字颜色为白色 */\n"
"    padding: 15px 32px;  /* 内边距 */\n"
"    text-align: center;  /* 文字居中 */\n"
"    text-decoration: none;\n"
"    display: inline-block;\n"
"    font-size: 16px;  /* 文字大小 */\n"
"    margin: 4px 2px;\n"
"    cursor: pointer;\n"
"    border-radius: 4px;  /* 圆角 */\n"
"}\n"
"\n"
"QPushButton#BeginDetBeginDet:hover {\n"
"    background-color: #45a049;  /* 鼠标悬停时的背景颜色 */\n"
"}\n"
"\n"
"QPushButton#StopDet {\n"
"    background-color: #f44336;  /* 设置背景颜色为红色 */\n"
"    border: none;  /* 无边框 */\n"
"    color: white;  /* 文字颜色为白色 */\n"
"    padding: 15px 32px;  /* 内边距 */\n"
"    text-align: center;  /* 文字居中 */\n"
"    text-decoration: none;\n"
"    display: inline-block;\n"
"    font-size: 16px;  /* 文字大小 */\n"
"    margin: 4px 2px;\n"
"    cursor: pointer;\n"
"    border-radius: 4px;  /* 圆角 */\n"
"}\n"
"\n"
"QPushButton#StopDet:hover {\n"
"    background-color: #d73828;  /* 鼠标悬停时的背景颜色 */\n"
"}\n"
"")
        self.StopDet.setAutoExclusive(False)
        self.StopDet.setObjectName("StopDet")
        self.gridLayout_2.addWidget(self.StopDet, 1, 0, 1, 1)
        self.BeginDet = QtWidgets.QPushButton(self.groupBox_3)
        self.BeginDet.setMinimumSize(QtCore.QSize(90, 25))
        self.BeginDet.setStyleSheet("QPushButton#BeginDet {\n"
"    background-color: #2FF000;  /* 设置背景颜色为绿色 */\n"
"    border: none;  /* 无边框 */\n"
"    color: white;  /* 文字颜色为白色 */\n"
"    padding: 15px 32px;  /* 内边距 */\n"
"    text-align: center;  /* 文字居中 */\n"
"    text-decoration: none;\n"
"    display: inline-block;\n"
"    font-size: 16px;  /* 文字大小 */\n"
"    margin: 4px 2px;\n"
"    cursor: pointer;\n"
"    border-radius: 4px;  /* 圆角 */\n"
"}\n"
"\n"
"QPushButton#BeginDet:hover {\n"
"    background-color: #4CAF50;  /* 鼠标悬停时的背景颜色 */\n"
"}\n"
"\n"
"QPushButton#StopDet {\n"
"    background-color: #f44336;  /* 设置背景颜色为红色 */\n"
"    border: none;  /* 无边框 */\n"
"    color: white;  /* 文字颜色为白色 */\n"
"    padding: 15px 32px;  /* 内边距 */\n"
"    text-align: center;  /* 文字居中 */\n"
"    text-decoration: none;\n"
"    display: inline-block;\n"
"    font-size: 16px;  /* 文字大小 */\n"
"    margin: 4px 2px;\n"
"    cursor: pointer;\n"
"    border-radius: 4px;  /* 圆角 */\n"
"}\n"
"\n"
"QPushButton#StopDet:hover {\n"
"    background-color: #d73828;  /* 鼠标悬停时的背景颜色 */\n"
"}\n"
"")
        self.BeginDet.setIconSize(QtCore.QSize(30, 30))
        self.BeginDet.setCheckable(True)
        self.BeginDet.setObjectName("BeginDet")
        self.gridLayout_2.addWidget(self.BeginDet, 0, 0, 1, 1)
        self.horizontalLayout_7.addWidget(self.groupBox_3)
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setMinimumSize(QtCore.QSize(350, 100))
        self.groupBox_4.setMaximumSize(QtCore.QSize(16777215, 300))
        self.groupBox_4.setStyleSheet("QGroupBox {\n"
"    font-size: 18px; /* 设置字体大小 */\n"
"    font-weight: bold; /* 字体加粗 */\n"
"    border: 1px solid #4CAF50; /* 设置边框为绿色 */\n"
"    border-radius: 10px; /* 设置圆角为10px */\n"
"    padding: 10px; /* 设置内边距为10px */\n"
"    margin-top: 10px; /* 设置顶部外边距为10px */\n"
"    background-color: #D3D3D3; /* 背景颜色为淡灰色 */\n"
"}\n"
"")
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout.setObjectName("gridLayout")
        self.resultWidget = QtWidgets.QListWidget(self.groupBox_4)
        self.resultWidget.setMinimumSize(QtCore.QSize(250, 60))
        self.resultWidget.setStyleSheet("QTextBrowser {\n"
"    background-color: #FAFAFA; /* 设置淡灰色背景 */\n"
"    color: #000000; /* 文字颜色设置为黑色 */\n"
"    font-size: 14px; /* 字体大小 */\n"
"    border: none; /* 无边框 */\n"
"    padding: 10px; /* 内边距 */\n"
"    box-shadow: 5px 5px 5px rgba(0, 0, 0, 0.2); /* 添加阴影效果 */\n"
"}\n"
"")
        self.resultWidget.setObjectName("resultWidget")
        self.gridLayout.addWidget(self.resultWidget, 1, 0, 1, 1)
        self.statistic_label = QtWidgets.QTextBrowser(self.groupBox_4)
        self.statistic_label.setMinimumSize(QtCore.QSize(250, 60))
        self.statistic_label.setStyleSheet("QTextBrowser {\n"
"    background-color: #FAFAFA; /* 设置淡灰色背景 */\n"
"    color: #000000; /* 文字颜色设置为黑色 */\n"
"    font-size: 14px; /* 字体大小 */\n"
"    border: none; /* 无边框 */\n"
"    padding: 10px; /* 内边距 */\n"
"    box-shadow: 5px 5px 5px rgba(0, 0, 0, 0.2); /* 添加阴影效果 */\n"
"}\n"
"")
        self.statistic_label.setObjectName("statistic_label")
        self.gridLayout.addWidget(self.statistic_label, 0, 0, 1, 1)
        self.horizontalLayout_7.addWidget(self.groupBox_4)
        self.horizontalLayout_7.setStretch(1, 1)
        self.horizontalLayout_7.setStretch(3, 1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setMinimumSize(QtCore.QSize(500, 350))
        self.groupBox_5.setStyleSheet("QGroupBox {\n"
"    font-size: 18px; /* 设置字体大小 */\n"
"    font-weight: bold; /* 字体加粗 */\n"
"    border: 2px solid #4CAF50; /* 设置边框为绿色，增加边框宽度 */\n"
"    border-radius: 10px; /* 设置圆角为10px */\n"
"    padding: 10px; /* 设置内边距为10px */\n"
"    margin-top: 10px; /* 设置顶部外边距为10px */\n"
"    background-color: #E8F5E9; /* 背景颜色为淡绿色 */\n"
"    color: #333333; /* 设置字体颜色为深灰色 */\n"
"}\n"
"")
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.raw_video = QtWidgets.QLabel(self.groupBox_5)
        self.raw_video.setMinimumSize(QtCore.QSize(450, 300))
        self.raw_video.setStyleSheet("background-image: url(:/png/icon/纯白图片.png);\n"
"QLabel {\n"
"    background-color: #FFFFFF; /* 设置背景颜色为白色 */\n"
"    border: 1px solid #CCCCCC; /* 设置边框为灰色 */\n"
"    border-radius: 5px; /* 设置圆角为5px */\n"
"    padding: 10px; /* 设置内边距为10px */\n"
"    text-align: center; /* 文字居中 */\n"
"    font-size: 14px; /* 字体大小 */\n"
"    color: #000000; /* 字体颜色为黑色 */\n"
"}\n"
"")
        self.raw_video.setText("")
        self.raw_video.setObjectName("raw_video")
        self.gridLayout_4.addWidget(self.raw_video, 0, 0, 1, 1)
        self.horizontalLayout_8.addWidget(self.groupBox_5)
        self.groupBox_7 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_7.setMinimumSize(QtCore.QSize(500, 350))
        self.groupBox_7.setStyleSheet("QGroupBox {\n"
"    font-size: 18px; /* 设置字体大小 */\n"
"    font-weight: bold; /* 字体加粗 */\n"
"    border: 2px solid #4CAF50; /* 设置边框为绿色，增加边框宽度 */\n"
"    border-radius: 10px; /* 设置圆角为10px */\n"
"    padding: 10px; /* 设置内边距为10px */\n"
"    margin-top: 10px; /* 设置顶部外边距为10px */\n"
"    background-color: #E8F5E9; /* 背景颜色为淡绿色 */\n"
"    color: #333333; /* 设置字体颜色为深灰色 */\n"
"}\n"
"")
        self.groupBox_7.setObjectName("groupBox_7")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_7)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.out_video = QtWidgets.QLabel(self.groupBox_7)
        self.out_video.setMinimumSize(QtCore.QSize(450, 300))
        self.out_video.setStyleSheet("background-image: url(:/png/icon/纯白图片.png);\n"
"QLabel {\n"
"    background-color: #FFFFFF; /* 设置背景颜色为白色 */\n"
"    border: 1px solid #CCCCCC; /* 设置边框为灰色 */\n"
"    border-radius: 5px; /* 设置圆角为5px */\n"
"    padding: 10px; /* 设置内边距为10px */\n"
"    text-align: center; /* 文字居中 */\n"
"    font-size: 14px; /* 字体大小 */\n"
"    color: #000000; /* 字体颜色为黑色 */\n"
"}\n"
"")
        self.out_video.setText("")
        self.out_video.setObjectName("out_video")
        self.gridLayout_5.addWidget(self.out_video, 0, 0, 1, 1)
        self.horizontalLayout_8.addWidget(self.groupBox_7)
        self.verticalLayout_2.addLayout(self.horizontalLayout_8)
        self.verticalLayout_2.setStretch(2, 2)
        self.verticalLayout_2.setStretch(3, 10)
        self.gridLayout_6.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1072, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "欢迎使用梨采摘识别系统"))
        self.groupBox_2.setTitle(_translate("MainWindow", "开始阶段"))
        self.file.setText(_translate("MainWindow", "文件"))
        self.camera.setText(_translate("MainWindow", "摄像头"))
        self.groupBox.setTitle(_translate("MainWindow", "iou/conf"))
        self.label_2.setText(_translate("MainWindow", "IoU"))
        self.label_3.setText(_translate("MainWindow", "置信度"))
        self.label_5.setText(_translate("MainWindow", "帧间延迟"))
        self.checkBox.setText(_translate("MainWindow", "启用"))
        self.groupBox_3.setTitle(_translate("MainWindow", "检测阶段"))
        self.StopDet.setText(_translate("MainWindow", "停止检测"))
        self.BeginDet.setText(_translate("MainWindow", "开始检测"))
        self.groupBox_4.setTitle(_translate("MainWindow", "状态显示"))
        self.groupBox_5.setTitle(_translate("MainWindow", "原图/原视频"))
        self.groupBox_7.setTitle(_translate("MainWindow", "检测图/检测视频"))
import img_rc
