import sys
from pathlib import Path
import time

# Add project root to sys.path so "from src.*" imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from src.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # splash_pix = QPixmap(r"C:\Users\M297802\Desktop\CT-PCD-GUI\src\splash.png")
    # splash = QSplashScreen(splash_pix)
    # splash.show()
    # app.processEvents()

    # # 2. Simulate time-consuming initialization
    # splash.showMessage(
    #     "Loading data...",
    #     Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
    #     Qt.GlobalColor.white,
    # )
    # app.processEvents()  # Update message
    # time.sleep(1)  # Simulate task

    # splash.showMessage(
    #     "Connecting to database...",
    #     Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
    #     Qt.GlobalColor.white,
    # )
    # app.processEvents()  # Update message
    # time.sleep(1)  # Simulate task
    window = MainWindow()
    # splash.finish(window)
    sys.exit(app.exec())
