from kivy.config import Config
from kivy.logger import Logger, LOG_LEVELS


## Kivy configuration
CONFIGURATION = {
    "kivy": {
        "window_icon": "graphics/icon.png",
        "exit_on_escape": False,
    },
    "graphics": {
        "width": 960,
        "height": 540,
        "resizable": False,
        "borderless": True,
    },
}
for category, settings in CONFIGURATION.items():
    for setting, value in settings.items():
        Config.set(category, setting, value)

## Logging configuration
# Logger.setLevel(LOG_LEVELS["debug"])  # For debugging
# Logger.setLevel(LOG_LEVELS["critical"])  # Disables logging

Logger.info(LOG_LEVELS)
