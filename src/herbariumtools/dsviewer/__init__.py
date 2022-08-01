import sys, os.path
import argparse

from PySide6.QtWidgets import QApplication

from .mainwindow import MainWindow

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("dsroot", help="root directory for the dataset", type=str, nargs="?", default="~/Projects/datasets/fgvc9-herbarium-2022")
    args = parser.parse_args()
    
    dsroot = os.path.expanduser(args.dsroot)
    
    app = QApplication()

    mw = MainWindow(dsroot)
    mw.show()
    
    sys.exit(app.exec_())

