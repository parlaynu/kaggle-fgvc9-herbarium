import math, re
import importlib

import torchvision.transforms as TF

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QLabel, QGridLayout, QSizePolicy

from .imagepanel import ImagePanel


def _get_class(class_fqn):
    class_path = class_fqn.split('.')
    class_name = class_path[-1]
    class_mpath = '.'.join(class_path[0:-1])
    
    class_module = importlib.import_module(class_mpath)

    return getattr(class_module, class_name)


class ImagesPanel(QWidget):
    
    splitChanged = Signal(str)
    categoryIdChanged = Signal(int)
    pageLayoutChanged = Signal(str)
    pageChanged = Signal(int)
    
    def __init__(self, image_source, parent=None):
        super().__init__(parent)
        #self.setStyleSheet("QWidget { background-color : red; }");
        
        self._isource = image_source
        self._layoutRe = re.compile(r'^(\d+)x(\d+)$')

        self._split = None
        self._category_id = -1
        self._num_pages = 1
        self._page = 0
        self._rows = 0
        self._cols = 0
        
        # build the grid and the image panels
        self._max_rows = 3
        self._max_cols = 5
        self._images = [[ImagePanel(self) for c in range(self._max_cols)] for r in range(self._max_rows)]
        
        self._ilayout = QGridLayout(self)
        for row, images in enumerate(self._images):
            for col, image in enumerate(images):
                image.show()
                self._ilayout.addWidget(image, row, col)
        
        # set the defaults
        self.setPageLayout("3x5", update=False)
        self.setPage(0, update=False)
        self.setCategoryId(-1, update=False)
        
        splits = self._isource.splits()
        self.setSplit(splits[0], update=True)
    
    def splits(self):
        return self._isource.splits()
    
    def setSplit(self, split, *, update=True):
        if self._split == split:
            return
        self._isource.set_split(split)
        self._split = split
        self.setPage(0, update=False)
        if update:
            self._updateImages()
        self.splitChanged.emit(self._split)

    def setCategoryId(self, category_id, *, update=True, up=True):
        category_id = int(category_id)
        if self._category_id == category_id:
            return
        self._isource.set_category_id(category_id, up=up)
        self._category_id = self._isource.category_id()
        self.setPage(0, update=False)
        if update:
            self._updateImages()
        self.categoryIdChanged.emit(self._category_id)

    def numPages(self):
        return self._num_pages
    
    def setPage(self, page, *, update=True):
        if self._page == page:
            return

        self._page = min(page, self._num_pages - 1)
        if update:
            self._updateImages()
        self.pageChanged.emit(self._page)

    def setPageLayout(self, layout, *, update=True):
        mo = self._layoutRe.match(layout)
        rows = int(mo.group(1))
        cols = int(mo.group(2))
        
        if self._rows == rows and self._cols == cols:
            return

        self._rows = min(rows, self._max_rows)
        self._cols = min(cols, self._max_cols)
        
        for row, images in enumerate(self._images):
            for col, image in enumerate(images):
                if row < self._rows and col < self._cols:
                    image.show()
                else:
                    image.hide()
        
        if update:
            self._updateImages()
        
        self.pageLayoutChanged.emit(layout)

    def _updateImages(self):
        if self._split is None or self._category_id is None:
            return
        if self._rows == 0 or self._cols == 0:
            return
        
        ipp = self._rows * self._cols
        num_images = self._isource.num_images()
        self._num_pages = math.ceil(num_images/ipp)
        if self._page >= self._num_pages:
            self.setPage(self._num_pages - 1, update=False)
        
        idx = self._page * self._rows * self._cols
        
        for row, images in enumerate(self._images):
            if row >= self._rows:
                break
            for col, image in enumerate(images):
                if col >= self._cols:
                    break
                
                iinfo = self._isource.image_info(idx)
                if iinfo is not None:
                    idata = self._isource.image(idx)
                    image.display(iinfo, idata)
                else:
                    image.clear()
                
                idx += 1


