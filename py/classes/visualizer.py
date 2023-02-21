from PyQt5.QtWidgets import QLabel, QMainWindow, QWidget, QVBoxLayout, QApplication
from PyQt5.QtGui import QPixmap


class Visualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualizer")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.label = QLabel(self)
        self.layout.addWidget(self.label)
        self.pixmap = QPixmap('../../static/mephi.png')
        QApplication.processEvents()
        self.show()

    def show_image(self, image_path):
        self.setWindowTitle(image_path)
        self.pixmap = QPixmap(image_path)
        self.label.setPixmap(self.pixmap)
        self.resize(self.pixmap.width(), self.pixmap.height())
        QApplication.processEvents()
