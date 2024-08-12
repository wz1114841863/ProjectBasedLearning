# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'client_3.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDialog, QGridLayout, QGroupBox,
    QLabel, QSizePolicy, QWidget)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(1366, 768)
        Dialog.setMinimumSize(QSize(1366, 768))
        Dialog.setMaximumSize(QSize(1366, 768))
        self.groupBox = QGroupBox(Dialog)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(10, 10, 1351, 751))
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.groupBox.setFont(font)
        self.groupBox.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.groupBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gridLayoutWidget = QWidget(self.groupBox)
        self.gridLayoutWidget.setObjectName(u"gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(0, 40, 1351, 711))
        self.gridLayout_2 = QGridLayout(self.gridLayoutWidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.label_pic_2 = QLabel(self.gridLayoutWidget)
        self.label_pic_2.setObjectName(u"label_pic_2")
        font1 = QFont()
        font1.setPointSize(24)
        font1.setBold(True)
        self.label_pic_2.setFont(font1)
        self.label_pic_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_2.addWidget(self.label_pic_2, 0, 1, 1, 1)

        self.label_pic_6 = QLabel(self.gridLayoutWidget)
        self.label_pic_6.setObjectName(u"label_pic_6")
        self.label_pic_6.setFont(font1)
        self.label_pic_6.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_2.addWidget(self.label_pic_6, 1, 2, 1, 1)

        self.label_pic_9 = QLabel(self.gridLayoutWidget)
        self.label_pic_9.setObjectName(u"label_pic_9")
        self.label_pic_9.setFont(font1)
        self.label_pic_9.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_2.addWidget(self.label_pic_9, 2, 2, 1, 1)

        self.label_pic_5 = QLabel(self.gridLayoutWidget)
        self.label_pic_5.setObjectName(u"label_pic_5")
        self.label_pic_5.setFont(font1)
        self.label_pic_5.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_2.addWidget(self.label_pic_5, 1, 1, 1, 1)

        self.label_pic_3 = QLabel(self.gridLayoutWidget)
        self.label_pic_3.setObjectName(u"label_pic_3")
        self.label_pic_3.setFont(font1)
        self.label_pic_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_2.addWidget(self.label_pic_3, 0, 2, 1, 1)

        self.label_pic_8 = QLabel(self.gridLayoutWidget)
        self.label_pic_8.setObjectName(u"label_pic_8")
        self.label_pic_8.setFont(font1)
        self.label_pic_8.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_2.addWidget(self.label_pic_8, 2, 1, 1, 1)

        self.label_pic_1 = QLabel(self.gridLayoutWidget)
        self.label_pic_1.setObjectName(u"label_pic_1")
        self.label_pic_1.setFont(font1)
        self.label_pic_1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_2.addWidget(self.label_pic_1, 0, 0, 1, 1)

        self.label_pic_4 = QLabel(self.gridLayoutWidget)
        self.label_pic_4.setObjectName(u"label_pic_4")
        self.label_pic_4.setFont(font1)
        self.label_pic_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_2.addWidget(self.label_pic_4, 1, 0, 1, 1)

        self.label_pic_7 = QLabel(self.gridLayoutWidget)
        self.label_pic_7.setObjectName(u"label_pic_7")
        self.label_pic_7.setFont(font1)
        self.label_pic_7.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_2.addWidget(self.label_pic_7, 2, 0, 1, 1)

        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(1060, 0, 201, 41))
        self.label_counter = QLabel(self.groupBox)
        self.label_counter.setObjectName(u"label_counter")
        self.label_counter.setGeometry(QRect(1240, 10, 81, 21))

        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.groupBox.setTitle(QCoreApplication.translate("Dialog", u"\u56fe\u7247\u504f\u597d\u6027\u6d4b\u8bd5", None))
        self.label_pic_2.setText(QCoreApplication.translate("Dialog", u"Display", None))
        self.label_pic_6.setText(QCoreApplication.translate("Dialog", u"Display", None))
        self.label_pic_9.setText(QCoreApplication.translate("Dialog", u"Display", None))
        self.label_pic_5.setText(QCoreApplication.translate("Dialog", u"Display", None))
        self.label_pic_3.setText(QCoreApplication.translate("Dialog", u"Display", None))
        self.label_pic_8.setText(QCoreApplication.translate("Dialog", u"Display", None))
        self.label_pic_1.setText(QCoreApplication.translate("Dialog", u"Display", None))
        self.label_pic_4.setText(QCoreApplication.translate("Dialog", u"Display", None))
        self.label_pic_7.setText(QCoreApplication.translate("Dialog", u"Display", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"\u8fd8\u9700\u70b9\u51fb\u6b21\u6570\uff1a", None))
        self.label_counter.setText(QCoreApplication.translate("Dialog", u"1", None))
    # retranslateUi

