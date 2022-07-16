from kivy.config import Config


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
