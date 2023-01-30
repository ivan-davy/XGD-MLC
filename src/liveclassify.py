# В данный блок можно добавлять модули, необходимые для работы Вашей программы
# import mymodule
                          ##
                         ##
                        #############
                         ##
                          ##

# В данный блок можно добавлять модули, необходимые для работы Вашей программы

import time
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import numpy as np
from struct import unpack


# В данный блок можно добавлять Ваши функции и классы
                          ##
                         ##
                        #############
                         ##
                          ##

# В данный блок можно добавлять Ваши функции и классы


class MyWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle('Emulator')
        self.vertical_layout = QVBoxLayout()
        self.setLayout(self.vertical_layout)
        self.start_button = QPushButton('Start')
        self.stop_button = QPushButton('Stop')
        self.label_time = QLabel('Time = 0')
        self.line_report = QLineEdit('Result = None')
        self.vertical_layout.addWidget(self.start_button)
        self.vertical_layout.addWidget(self.stop_button)
        self.vertical_layout.addWidget(self.label_time)
        self.vertical_layout.addWidget(self.line_report)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.timeout.connect(self.periodic_function)
        self.start_button.clicked.connect(self.on_click_start_button)
        self.stop_button.clicked.connect(self.on_click_stop_button)
        self.data_x = np.arange(0, 4096, 1, dtype='float64')
        self.data_y = np.zeros(4096)
        self.A_coef = 1.0
        self.B_coef = 0.0
        self.rTime = 0.0
        self.lTime = 0.0

        self.start_time = time.time()
        self.cps = np.ones(4096)
        self.dead_time_percent = 0.0

        # Здесь можно поменять файл спектра, 
        # на основе которого будет генерироваться информация 
        self.open_sps_file('Сs137_15cm.sps')

    def on_click_start_button(self):
        self.start_time = time.time()
        self.timer.start(1000) # Здесь можно поменять интерал времени обновления результата (аргумент в милисекундах)

    def on_click_stop_button(self):
        self.update_data()
        self.label_time.setText('Time = ' + str(round(self.rTime,2)))
        self.timer.stop()

    def open_sps_file(self, filename):
        raw_file = open(filename, 'rb')
        raw_file.seek(301)
        raw_data = raw_file.read(4)
        self.lTime = unpack("I", raw_data)[0]
        raw_data = raw_file.read(4)
        self.rTime = unpack("I", raw_data)[0]
        self.dead_time_percent = 1.0 - (self.lTime/self.rTime)
        raw_file.seek(440)
        raw_data = raw_file.read(4)
        self.A_coef = unpack("f", raw_data)[0]
        raw_data = raw_file.read(4)
        self.B_coef = unpack("f", raw_data)[0]
        raw_file.seek(1024)
        raw_data = raw_file.read(4*4096)
        raw_file.close()
        self.cps = unpack('4096I', raw_data)
        self.cps = np.array(self.cps)
        self.cps = self.cps/self.lTime



    def update_data(self):
        self.rTime = time.time() - self.start_time
        self.lTime = self.rTime * (1.0 - self.dead_time_percent)
        self.data_y = self.cps * self.lTime


    def periodic_function(self):
        # в данный блок нужно добавить вызовы Ваших функций и
        # результат сохранить в переменной your_function_result
                          ##
                         ##
                        #############
                         ##
                          ##
        your_function_result = (True, 'Источник') # здесь происходит присвоение (результат True или False)
        #your_function_result = (False, 'Фон')
        # в данный блок нужно добавить Вашу функцию и результат
        self.label_time.setText('Time = ' + str(round(self.rTime,2)))
        self.line_report.setText('Result = ' + str(your_function_result))



if __name__ == '__main__':
    app = QApplication([])
    MW = MyWidget()
    MW.show()
    app.exec_()