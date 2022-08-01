import math

from PySide6.QtCore import Slot, Signal

from PySide6.QtGui import Qt, QValidator, QTransform, QIcon

from PySide6.QtWidgets import QWidget, QLabel, QComboBox, QLineEdit, QPushButton, QCheckBox
from PySide6.QtWidgets import QHBoxLayout, QGridLayout, QStyle


class _IndexValidator(QValidator):
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def validate(self, value, pos):
        try:
            int(value)
            return QValidator.Acceptable
        except ValueError:
            return QValidator.Invalid


class ControlPanel(QWidget):

    def __init__(self, images_panel, parent=None):
        super().__init__(parent)
        self._ipanel = images_panel

        self._ipanel.splitChanged.connect(self.splitChanged)
        self._ipanel.categoryIdChanged.connect(self.categoryIdChanged)
        self._ipanel.pageChanged.connect(self.pageChanged)

        self._build()


    @Slot(str)
    def splitChanged(self, value):
        idx = self._split.findText(value)
        self._split.setCurrentIndex(idx)
    
    @Slot(int)
    def categoryIdChanged(self, value):
        self._category_id.setText(str(value))
    
    @Slot(int)
    def pageChanged(self, value):
        self._pageW.setText(str(value))
    
    @Slot(str)
    def pageLayoutChanged(self, value):
        idx = self._split.findText(value)
        self._split.setCurrentIndex(idx)
        
    @Slot(int)
    def _setSplit(self, value):
        split = self._split.currentText()
        self._ipanel.setSplit(split)
    
    @Slot()
    def _setCategoryId(self):
        self._doSetCategoryId(True)
    
    @Slot()
    def _prvCategoryId(self):
        category_id = int(self._category_id.text()) - 1
        self._category_id.setText(str(category_id))
        self._doSetCategoryId(False)
    
    @Slot()
    def _nxtCategoryId(self):
        category_id = int(self._category_id.text()) + 1
        self._category_id.setText(str(category_id))
        self._doSetCategoryId(True)
    
    def _doSetCategoryId(self, up):
        category_id = int(self._category_id.text())
        self._ipanel.setCategoryId(category_id, up=up)

    @Slot(str)
    def _setPageLayout(self, value):
        layout = self._pageLayout.currentText()
        self._ipanel.setPageLayout(layout)

    @Slot()
    def _skipTo(self):
        page = max(0, int(self._pageW.text()))
        self._pageW.setText(str(page))
        self._ipanel.setPage(page)
    
    @Slot()
    def _stepRev(self):
        page = max(0, int(self._pageW.text()) - 1)
        self._pageW.setText(str(page))
        self._ipanel.setPage(page)
    
    @Slot()
    def _stepFwd(self):
        page = int(self._pageW.text()) + 1
        self._pageW.setText(str(page))
        self._ipanel.setPage(page)
        
    @Slot()
    def _skipRev(self):
        page = max(0, int(self._pageW.text()) - 5)
        self._pageW.setText(str(page))
        self._ipanel.setPage(page)
    
    @Slot()
    def _skipFwd(self):
        page = int(self._pageW.text()) + 5
        self._pageW.setText(str(page))
        self._ipanel.setPage(page)
        
    @Slot()
    def _skipStart(self):
        page = 0
        self._pageW.setText(str(page))
        self._ipanel.setPage(page)
    
    @Slot()
    def _skipEnd(self):
        page = self._ipanel.numPages() - 1
        self._pageW.setText(str(page))
        self._ipanel.setPage(page)
        
    def _build(self):
        glayout = QGridLayout()
        
        # split selection
        self._split = widget = QComboBox(self)
        for s in self._ipanel.splits():
            widget.addItem(s)
        
        label = QLabel(self)
        label.setText("Split")
        label.setBuddy(widget)
        
        glayout.addWidget(label, 0, 0)
        glayout.addWidget(widget, 0, 1)
        
        widget.currentIndexChanged.connect(self._setSplit)
        
        # category_id selection
        self._category_id = widget = QLineEdit(self)
        widget.setText("-1")
        widget.setAlignment(Qt.AlignCenter)

        validator = _IndexValidator(self)
        widget.setValidator(validator)
        
        prvPx = self.style().standardIcon(QStyle.SP_MediaPlay).pixmap(32)
        prvPx = prvPx.transformed(QTransform().rotate(180))
        prvIcon = QIcon(prvPx)
        prv = QPushButton(prvIcon, "", self)

        nxtIcon = self.style().standardIcon(QStyle.SP_MediaPlay)
        nxt = QPushButton(nxtIcon, "", self)
        
        prv.clicked.connect(self._prvCategoryId)
        nxt.clicked.connect(self._nxtCategoryId)
        
        layout = QHBoxLayout()
        layout.addWidget(prv)
        layout.addWidget(widget)
        layout.addWidget(nxt)
        
        label = QLabel(self)
        label.setText("Category")
        label.setBuddy(widget)
        
        glayout.addWidget(label, 1, 0)
        glayout.addLayout(layout, 1, 1)
        
        widget.editingFinished.connect(self._setCategoryId)


        # layout selection
        self._pageLayout = widget = QComboBox(self)
        widget.addItem("1x1")
        widget.addItem("1x2")
        widget.addItem("2x3")
        widget.addItem("3x5")
        widget.setCurrentIndex(3)
        
        label = QLabel(self)
        label.setText("Layout")
        label.setBuddy(widget)
        
        widget.currentIndexChanged.connect(self._setPageLayout)
        
        pageLayout = QHBoxLayout()
        pageLayout.addWidget(label)
        pageLayout.addWidget(widget)

        # the page controls
        pageControls = self._build_pagecontrols()
        
        # layout the panel
        layout = QHBoxLayout(self)
        layout.addLayout(glayout)
        layout.addStretch(1)
        layout.addLayout(pageControls)
        layout.addStretch(1)
        layout.addLayout(pageLayout)
        
    def _build_pagecontrols(self):
        
        self._pageW = QLineEdit(self)
        self._pageW.setAlignment(Qt.AlignCenter)
        
        validator = _IndexValidator(self)
        self._pageW.setValidator(validator)
        
        self._pageW.setText("0")
        self._pageW.editingFinished.connect(self._skipTo)

        startIcon = self.style().standardIcon(QStyle.SP_MediaSkipBackward)
        start = QPushButton(startIcon, "", self)
        
        pprvIcon = self.style().standardIcon(QStyle.SP_MediaSeekBackward)
        pprv = QPushButton(pprvIcon, "", self)

        prvPx = self.style().standardIcon(QStyle.SP_MediaPlay).pixmap(32)
        prvPx = prvPx.transformed(QTransform().rotate(180))
        prvIcon = QIcon(prvPx)
        prv = QPushButton(prvIcon, "", self)

        nxtIcon = self.style().standardIcon(QStyle.SP_MediaPlay)
        nxt = QPushButton(nxtIcon, "", self)
        
        nnxtIcon = self.style().standardIcon(QStyle.SP_MediaSeekForward)
        nnxt = QPushButton(nnxtIcon, "", self)

        endIcon = self.style().standardIcon(QStyle.SP_MediaSkipForward)
        end = QPushButton(endIcon, "", self)

        start.clicked.connect(self._skipStart)
        pprv.clicked.connect(self._skipRev)
        prv.clicked.connect(self._stepRev)
        nxt.clicked.connect(self._stepFwd)
        nnxt.clicked.connect(self._skipFwd)
        end.clicked.connect(self._skipEnd)
        
        layout = QHBoxLayout()
        layout.addWidget(start)
        layout.addWidget(pprv)
        layout.addWidget(prv)
        layout.addWidget(self._pageW)
        layout.addWidget(nxt)
        layout.addWidget(nnxt)
        layout.addWidget(end)
        
        return layout

