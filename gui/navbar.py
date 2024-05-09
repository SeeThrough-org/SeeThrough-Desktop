from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QSpacerItem, QSizePolicy, QTabWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize

class NavBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(64)

        # Create main layout
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)  # Reduced margins to 10

        # Create logo label
        logo = QLabel("SeeThrough")
        logo.setStyleSheet("font-family: Montserrat; font-size: 20px; font-weight: bold; color: #333;")
        layout.addWidget(logo)

        # Add left spacer
        left_spacer = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addItem(left_spacer)

        # Create tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setFixedSize(400, 40)  # Set a fixed size for the QTabWidget

        self.tab_widget.setStyleSheet(
            """
            QTabWidget::pane { border: none; background-color: #f7f7f7; }
            QTabBar::tab { background-color: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 10px 20px; font-size: 11px; }
            QTabBar::tab:hover { background-color: #f2f2f2; }
            QTabBar::tab:selected { background-color: #007ACC; color: #fff; }
            """
        )
        self.realtime_tab = QWidget()
        self.static_tab = QWidget()
        self.video_tab = QWidget()
        self.tab_widget.addTab(self.realtime_tab, "Realtime Dehazing")
        self.tab_widget.addTab(self.static_tab, "Image Dehazing")
        self.tab_widget.addTab(self.video_tab, "Video Dehazing")
        layout.addWidget(self.tab_widget)

        # Add right spacer
        right_spacer = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addItem(right_spacer)

        # Create exit button
        self.exit_button = QPushButton()
        self.exit_button.setIcon(QIcon("assets/exit.svg"))
        self.exit_button.setIconSize(QSize(32, 32))
        self.exit_button.setStyleSheet("background-color: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 5px;")
        layout.addWidget(self.exit_button)

        self.setLayout(layout)