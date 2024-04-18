from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QHBoxLayout,
    QLabel,
  
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize


class NavBar(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setFixedHeight(64)

        # Label for the logo
        logo = QLabel("seeThrough")
        logo.setStyleSheet(
            """
            QLabel {
                font-family: "Montserrat";
                font-size: 20px;
                font-weight: bold;
                color: #191919;
            }
        """
        )

        # Create buttons for frame switching
        self.realtime_button = QPushButton("Realtime Dehazing")
        self.realtime_button.setObjectName("realtime_button")
        self.realtime_button.setStyleSheet(
            """
            QPushButton {
                background-color: #fff;
                border: 1px solid gray;
                border-radius: 10px;
                padding: 10px 60px;
                font-size: 13px;
            }

            QPushButton:hover {
                background-color: #373030;
                color: #fff;
            }
        """
        )

        self.static_button = QPushButton("Image Dehazing")
        self.static_button.setObjectName("static_button")
        self.static_button.setStyleSheet(
            """
            QPushButton {
                background-color: #fff;
                border: 1px solid gray;
                border-radius: 10px;
                padding: 10px 60px;
                font-size: 13px;
            }

            QPushButton:hover {
                background-color: #373030;
                color: #fff;
            }
        """
        )

        self.video_button = QPushButton("Video Dehazing")
        self.video_button.setStyleSheet(
            """
            QPushButton {
                background-color: #fff;
                border: 1px solid gray;
                border-radius: 10px;
                padding: 10px 60px;
                font-size: 13px;
            }

            QPushButton:hover {
                background-color: #373030;
                color: #fff;
            }
        """
        )

        self.exit_button = QPushButton()
        self.exit_button.setIcon(QIcon("./gui/assets/icons/exit.svg"))
        self.exit_button.setIconSize(QSize(32, 32))
        self.exit_button.setStyleSheet(
            """
            QPushButton {
                background-color: #fff;
                border: 1px solid gray;
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #eeeeee;
            }
        """
        )

        # Add buttons to the navbar
        layout = QHBoxLayout(self)
        layout.addWidget(logo, alignment=Qt.AlignLeft)
        layout.addWidget(self.realtime_button)
        layout.addWidget(self.static_button)
        layout.addWidget(self.video_button)
        layout.addWidget(self.exit_button, alignment=Qt.AlignRight)