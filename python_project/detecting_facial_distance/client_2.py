# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'client_2.ui'
##
## Created by: Qt User Interface Compiler version 6.6.2
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
from PySide6.QtWidgets import (QApplication, QDialog, QGroupBox, QLabel,
    QSizePolicy, QWidget)

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
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setLayoutDirection(Qt.LeftToRight)
        self.groupBox.setAlignment(Qt.AlignCenter)
        self.label_pic = QLabel(self.groupBox)
        self.label_pic.setObjectName(u"label_pic")
        self.label_pic.setGeometry(QRect(9, 20, 1331, 720))
        font1 = QFont()
        font1.setPointSize(24)
        self.label_pic.setFont(font1)
        self.label_pic.setAlignment(Qt.AlignCenter)
        self.label_time = QLabel(self.groupBox)
        self.label_time.setObjectName(u"label_time")
        self.label_time.setGeometry(QRect(480, 690, 401, 40))
        self.label_time.setFont(font1)
        self.label_time.setAlignment(Qt.AlignCenter)
        self.label_warning = QLabel(self.groupBox)
        self.label_warning.setObjectName(u"label_warning")
        self.label_warning.setGeometry(QRect(440, 30, 471, 40))
        self.label_warning.setFont(font1)
        self.label_warning.setAlignment(Qt.AlignCenter)

        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.groupBox.setTitle(QCoreApplication.translate("Dialog", u"\u56fe\u7247\u5c55\u793a", None))
        self.label_pic.setText(QCoreApplication.translate("Dialog", u"Display", None))
        self.label_time.setText(QCoreApplication.translate("Dialog", u"\u6301\u7eed\u65f6\u95f4", None))
        self.label_warning.setText(QCoreApplication.translate("Dialog", u"\u8b66\u544a\u8bed", None))
    # retranslateUi

