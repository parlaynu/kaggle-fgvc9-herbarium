from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel

from .datasource import DataSource
from .imagespanel import ImagesPanel
from .controlpanel import ControlPanel


class MainWindow(QMainWindow):
    def __init__(self, dsroot, parent=None, flags=Qt.WindowFlags()):
        super().__init__(parent, flags)
        
        self.setWindowTitle("Dataset Explorer")
        
        ssize = QApplication.instance().primaryScreen().availableSize()
        #width = int(ssize.width() * 0.9)
        #height = int(0.85 * width / 1920 * 1080)
        
        self.resize(ssize.width(), ssize.height())
        
        central = QLabel(self)
        #central.setStyleSheet("QLabel { background-color : red; }");

        dsource = DataSource(dsroot, self)
        ip = ImagesPanel(dsource, central)
        cp = ControlPanel(ip, central)
        
        layout = QVBoxLayout()
        layout.addWidget(cp)
        layout.addWidget(ip, stretch=2)
        central.setLayout(layout)

        self.setCentralWidget(central)

