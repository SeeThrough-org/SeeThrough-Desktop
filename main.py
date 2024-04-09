from gui.gui import GUI
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QStackedWidget, QSizePolicy,
                             QGridLayout, QHBoxLayout, QMessageBox, QLabel, QMenu, QDialog, QWidget)


def main():
    app = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
