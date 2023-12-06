import sys
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (QApplication, QSplashScreen, QLabel, QProgressBar, QVBoxLayout, QWidget, QSizePolicy,
                             QDesktopWidget)
from PyQt5.QtGui import QPixmap
import gui as gui


class SplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__(QPixmap('E:/School/Thesis/Realtime-Dehazing/gui/assets/try.jpg'))
        self.setWindowFlags(Qt.SplashScreen)
        self.showMessage("Loading...", Qt.AlignHCenter |
                         Qt.AlignBottom, Qt.white)
        self.show()

    def display_splash_screen():
        app = QApplication([])

        # Load the image from the file path
        try:
            background_pixmap = QPixmap(
                'E:/School/Thesis/Realtime-Dehazing/gui/assets/try.jpg')
        except FileNotFoundError:
            print("Error: Image file not found")
            sys.exit(1)

        if background_pixmap.isNull():
            print("Error: Unable to load image")
            sys.exit(1)

        # Create a QSplashScreen with the resized pixmap image
        splash = QSplashScreen(background_pixmap)
        # Increase the size of the splash screen
        splash.setPixmap(background_pixmap.scaled(
            800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Desktop Screen
        desktop = app.desktop()
        screen_rect = desktop.availableGeometry(desktop.primaryScreen())

        # Center SplashScreen upon launch
        splash_width = 800
        splash_height = 400
        x = (screen_rect.width() - splash_width) // 2
        y = (screen_rect.height() - splash_height) // 2
        splash.setGeometry(x, y, splash_width, splash_height)

        # Placeholders for widgets
        main_widget = QWidget(splash)
        layout = QVBoxLayout(main_widget)
        main_widget.setLayout(layout)

        # LOGO WIDGET
        logo = QLabel("seeThrough", main_widget)
        logo.setStyleSheet("""
        QLabel {
            color: #323232;
            font-size: 50px;
            font-weight: bold;
        }""")

        # PROGRESS LABEL WIDGET
        progress_label = QLabel(main_widget)
        progress_label.setStyleSheet("""
        QLabel {
            color: #323232;
            font-size: 25px;
            font-family: "Segoe UI";
        }""")
        progress_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Add widgets to layout
        layout.addWidget(logo)
        layout.addWidget(progress_label)

        splash.show()

        # Function to update the progress bar
        def update_progress():
            nonlocal i
            if i <= 100:
                progress_label.setText(f"Loading... {i}%")
                i += 1
            else:
                splash.close()
                # Once the splash screen closes, create and show the main window
                gui = GUI()
                gui.show()

        i = 0
        timer = QTimer()
        timer.timeout.connect(update_progress)
        timer.start(40)  # Update every 40 milliseconds

        # Close splash screen on application exit
        app.aboutToQuit.connect(splash.close)

        sys.exit(app.exec_())
