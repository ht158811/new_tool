from enum import Enum


class Theme(str, Enum):
    LIGHT = "light"
    DARK = "dark"


LIGHT_STYLESHEET = """
QWidget { background: #f5f7fa; color: #202631; font-size: 14px; }
QFrame#sidebar { background: #ffffff; border-right: 1px solid #dfe3e8; }
QPushButton { border: 0; border-radius: 7px; padding: 10px 14px; text-align: left; }
QPushButton:hover { background: #eef2f7; }
QPushButton:checked { background: #dceaff; color: #1558b0; font-weight: 600; }
QPushButton#primary { background: #2369d8; color: white; text-align: center; }
QFrame#card { background: white; border: 1px solid #e0e4ea; border-radius: 10px; }
QLineEdit { background: white; border: 1px solid #cfd5dd; border-radius: 6px; padding: 8px; }
QListWidget { background: white; border: 1px solid #dfe3e8; border-radius: 8px; }
"""

DARK_STYLESHEET = """
QWidget { background: #171a1f; color: #e9edf2; font-size: 14px; }
QFrame#sidebar { background: #20242b; border-right: 1px solid #343a44; }
QPushButton { border: 0; border-radius: 7px; padding: 10px 14px; text-align: left; }
QPushButton:hover { background: #2a3039; }
QPushButton:checked { background: #23456f; color: #8fc1ff; font-weight: 600; }
QPushButton#primary { background: #347bdc; color: white; text-align: center; }
QFrame#card { background: #20242b; border: 1px solid #343a44; border-radius: 10px; }
QLineEdit { background: #20242b; border: 1px solid #4a5260; border-radius: 6px; padding: 8px; }
QListWidget { background: #20242b; border: 1px solid #343a44; border-radius: 8px; }
"""


def stylesheet(theme: Theme) -> str:
    return DARK_STYLESHEET if theme is Theme.DARK else LIGHT_STYLESHEET

