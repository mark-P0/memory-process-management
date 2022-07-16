import modules.scripts.config  # Configs first before Kivy app
from modules.scripts.core import DynamicType

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


class PartitionedDynamic(MDApp):
    '''
    `MDApp`-derived class automatically imports similarly named `.kv` file,
    i.e. `partitioneddynamic.kv`. Case-insensitive.
    '''

    title = "Memory Management: Partitioned Dynamic (First Fit & Best Fit)"

    ## Process properties
    jobs: list[list]
    sizes: list[int]
    dynamic_type: DynamicType
    enable_compaction: bool

    job_table: pd.DataFrame
    events: pd.DataFrame
    summary_table: pd.DataFrame
    timepoints: list
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
        # figure.close()

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
    inst = PartitionedDynamic()

    inst.run()
