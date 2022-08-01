import io
from PIL import Image

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from PySide6.QtWidgets import QVBoxLayout, QSizePolicy
from PySide6.QtWidgets import QWidget, QLabel

class ImagePanel(QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        #self.setStyleSheet("QWidget { background-color : red; }");
        #self._image.setStyleSheet("QLabel { background-color : red; }");
        #self._title.setStyleSheet("QLabel { background-color : red; }");
        
        sp = self.sizePolicy()
        self.setSizePolicy(sp)

        self._idata = None

        self._image = QLabel(self)
        self._image.setAlignment(Qt.AlignCenter)
        
        sp = self._image.sizePolicy()
        sp.setHorizontalPolicy(QSizePolicy.Ignored)
        sp.setVerticalPolicy(QSizePolicy.Ignored)
        self._image.setSizePolicy(sp)
        
        self._title = QLabel(self)
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        self._title.setText("")
        
        layout = QVBoxLayout()
        layout.addWidget(self._image, stretch=2)
        layout.addWidget(self._title)
        self.setLayout(layout)
        
    def display(self, iinfo, idata):
        self._iinfo = iinfo
        self._idata = idata
        
        self.redisplay()

    def redisplay(self):
        if self.isHidden() or self._idata is None:
            return
            
        image = self._idata
        
        iw = image.width
        ih = image.height
        iname = self._iinfo['image_name']
        
        cid = self._iinfo['category_id']
        cname = self._iinfo['label']
        
        self._title.setText(f"{iname}\n{iw}x{ih}\n{cid} - {cname}")

        # jump through some hoops to get a qimage
        imageB = io.BytesIO()
        image.save(imageB, format="jpeg")
        imageB.seek(0)
        qimage = QImage.fromData(imageB.getvalue())
        
        pixmap = QPixmap.fromImage(qimage)
        
        w = min(self._image.width(), iw)
        h = min(self._image.height(), ih)
        
        if w == 0 or h == 0:
            self._image.setPixmap(pixmap)
        else:
            self._image.setPixmap(pixmap.scaled(w, h, Qt.KeepAspectRatio))
        
    def clear(self):
        self._title.clear()
        self._image.clear()
        self._idata = None
        
    def resizeEvent(self, event):
        self.redisplay()
