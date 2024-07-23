from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QSize, pyqtSignal

class NavBar(QWidget):
    tab_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(70) 


        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#ffffff"))
        self.setPalette(palette)

        layout = QHBoxLayout()
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(15)

        logo = QLabel("SeeThrough")
        logo_font = QFont("Arial", 18, QFont.Bold)
        logo.setFont(logo_font)
        logo.setStyleSheet("color: #333333;")
        layout.addWidget(logo)

        layout.addStretch(1)

        self.nav_buttons = []
        nav_items = ["Realtime Dehazing", "Image Dehazing", "Video Dehazing"]
        for index, item in enumerate(nav_items):
            button = QPushButton(item)
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            button.setFont(QFont("Arial", 11))
            button.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: 1px solid #cccccc;
                    color: #555555;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #f0f0f0;
                }
                QPushButton:checked {
                    background-color: #e1f5fe;
                    color: #0288d1;
                    font-weight: bold;
                    border-color: #0288d1;
                }
            """)
            button.clicked.connect(lambda checked, idx=index: self.tab_changed.emit(idx))
            self.nav_buttons.append(button)
            layout.addWidget(button)

      
        self.nav_buttons[0].setChecked(True)

        layout.addStretch(1)

        self.exit_button = QPushButton()
        self.exit_button.setIcon(QIcon("assets/exit.svg"))
        self.exit_button.setIconSize(QSize(24, 24))
        self.exit_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.exit_button.setFixedSize(40, 40)
        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #ffebee;
            }
            QPushButton:pressed {
                background-color: #ffcdd2;
            }
        """)
        layout.addWidget(self.exit_button)

        self.setLayout(layout)

    def set_active_tab(self, index):
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjustButtonsSize()

    def adjustButtonsSize(self):
        available_width = self.width() - 400  
        button_width = max(120, available_width // 3)  
        for button in self.nav_buttons:
            button.setFixedWidth(button_width)