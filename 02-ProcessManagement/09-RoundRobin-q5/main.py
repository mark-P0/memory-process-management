from modules.scripts import config, core  # Configs first before Kivy app

from kivymd.app import MDApp
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.dialog import MDDialog

import pandas as pd
from modules.widgets.screens import (
    HomeScreen,
    LoadingScreen,
    OutputScreen,
    SummaryScreen,
)


class RoundRobin(MDApp):
    '''
    `MDApp`-derived class automatically imports similarly named `.kv` file,
    i.e. `roundrobin.kv`. Case-insensitive.
    '''

    title = "Process Management: Round Robin (q=5)"
    scheduling = core.SchedulingType.RR

    ## Process properties
    inputs: list[list]
    quantum: int

    characteristic_table: pd.DataFrame
    CPU_util: float
    ATA: float
    AWT: float
    events: pd.DataFrame
    timepoints: list[list]
    plot: core.Plotting
    figures: list[tuple]

    current_screens_base: str = ""  # Base name of output screens
    current_output_screen: str  # Name of current output screen

    ## TODO: Define title texts
    dialog_title_map = {
        "info": "",
        "warning": "Heads up!",
        "error": "Uh-oh...!",
    }

    def on_start(self):
        self.theme_cls.primary_palette = "Gray"

    # def on_stop(self):
    #     raise SystemExit

    def popup_message(self, title_type, message):
        title = self.dialog_title_map.get(title_type, "")

        dialog = MDDialog(
            # type="confirmation",
            title=title,
            text=message,
        )
        dialog.open()


if __name__ == "__main__":
    inst = RoundRobin()

    inst.run()
