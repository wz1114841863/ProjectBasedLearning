# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'client_1.ui'
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
from PySide6.QtWidgets import (QApplication, QGroupBox, QPushButton, QSizePolicy,
    QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(458, 265)
        font = QFont()
        font.setPointSize(18)
        Form.setFont(font)
        self.groupBox_comStatement_2 = QGroupBox(Form)
        self.groupBox_comStatement_2.setObjectName(u"groupBox_comStatement_2")
        self.groupBox_comStatement_2.setGeometry(QRect(10, 150, 461, 91))
        font1 = QFont()
        font1.setPointSize(12)
        self.groupBox_comStatement_2.setFont(font1)
        self.pushButton_start_test = QPushButton(self.groupBox_comStatement_2)
        self.pushButton_start_test.setObjectName(u"pushButton_start_test")
        self.pushButton_start_test.setGeometry(QRect(20, 40, 156, 29))
        self.pushButton_clear = QPushButton(self.groupBox_comStatement_2)
        self.pushButton_clear.setObjectName(u"pushButton_clear")
        self.pushButton_clear.setGeometry(QRect(230, 40, 156, 29))
        self.groupBox_comStatement_3 = QGroupBox(Form)
        self.groupBox_comStatement_3.setObjectName(u"groupBox_comStatement_3")
        self.groupBox_comStatement_3.setGeometry(QRect(10, 40, 461, 91))
        self.groupBox_comStatement_3.setFont(font1)
        self.pushButton_start_prog = QPushButton(self.groupBox_comStatement_3)
        self.pushButton_start_prog.setObjectName(u"pushButton_start_prog")
        self.pushButton_start_prog.setGeometry(QRect(20, 40, 156, 29))
        self.pushButton_stop_prog = QPushButton(self.groupBox_comStatement_3)
        self.pushButton_stop_prog.setObjectName(u"pushButton_stop_prog")
        self.pushButton_stop_prog.setGeometry(QRect(230, 40, 156, 29))

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.groupBox_comStatement_2.setTitle(QCoreApplication.translate("Form", u"\u56fe\u7247\u504f\u597d\u6027\u6d4b\u8bd5", None))
        self.pushButton_start_test.setText(QCoreApplication.translate("Form", u"\u5f00\u59cb", None))
        self.pushButton_clear.setText(QCoreApplication.translate("Form", u"\u91cd\u7f6e", None))
        self.groupBox_comStatement_3.setTitle(QCoreApplication.translate("Form", u"\u7a0b\u5e8f\u8fd0\u884c", None))
        self.pushButton_start_prog.setText(QCoreApplication.translate("Form", u"\u5f00\u59cb", None))
        self.pushButton_stop_prog.setText(QCoreApplication.translate("Form", u"\u7ed3\u675f", None))
    # retranslateUi

