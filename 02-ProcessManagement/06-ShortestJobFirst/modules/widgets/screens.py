## TODO: Put classes into separate files?
## TODO: Move "constants" to separate file?

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label

from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.scatter import Scatter, Matrix
from kivy.uix.image import Image, AsyncImage
from kivy.core.image import Image as CoreImage
from kivy.uix.widget import Widget
from kivy.clock import Clock, mainthread  # GUI changes must be done on the main thread

from kivymd.app import MDApp
from kivymd.uix.textfield import MDTextField
from kivymd.uix.dialog import MDDialog

import threading
import random
from string import ascii_letters

import pandas as pd
from ..scripts import core

## Home screen


class TableLabel(Label):
    ...


class TableLabelHeader(TableLabel):
    ...


class TableInput(MDTextField):
    def filter_func(self, new_character, is_from_undo):
        if not new_character.isdigit():
            new_character = ""

        return new_character

    def filter_func_time_hour(self, new_character, is_from_undo):
        actual = (
            (new_character + self.text)
            if self.cursor_index() == 0
            else (self.text + new_character)
        )

        if not (new_character.isdigit() and int(actual) <= 12):
            new_character = ""

        # if actual[0] != "0" and int(actual) < 10:
        #     new_character = "0" + new_character

        return new_character

    def filter_func_time_minutes(self, new_character, is_from_undo):
        ## TODO: Refactor this lol
        actual = (
            (new_character + self.text)
            if self.cursor_index() == 0
            else (self.text + new_character)
        )

        if not (new_character.isdigit() and int(actual) < 60):
            new_character = ""

        # if actual[0] != "0" and int(actual) < 10:
        #     new_character = "0" + new_character

        # if int(actual) < 10:
        #     if actual[0] == "0":
        #         ...
        #     else:
        #         new_character = "0" + new_character

        return new_character

    def unfocus_add_zero(self):
        if (
            not self.focus
            and self.text.isnumeric()
            and self.text[0] != "0"
            and int(self.text) < 10
        ):
            self.text = "0" + self.text


class TableInputTime(BoxLayout):
    @property
    def text(self):
        texts = [child.text for child in reversed(self.children)]
        texts[-1] = " " + texts[-1]  # Add leading space to period indicator
        return "".join(texts)

    @text.setter
    def text(self, value):
        """
        `value` format expected to be
        HH:MM PP
        """

        value = value.replace(":", " ")
        hh, mm, ampm = value.split(" ")

        self.ids["hour"].text = hh
        self.ids["minute"].text = mm
        self.ids["period"].text = ampm


class HomeScreen(Screen):
    partition_name_template = "P{}"

    def on_kv_post(self, *args, **kwargs):
        super().on_kv_post(*args, **kwargs)

        self._create_table()

        # ## DEBUG: Add sample values
        # self._add_table_values()

    def _create_table(self, rows=5):
        table = self.ids["proc_table"]

        ## Add headers
        headers = ("Process", "Burst Time", "Arrival Time")

        header = headers[0]
        new_label = TableLabelHeader(text=header, bold=True, size_hint_x=0.5)
        table.add_widget(new_label)

        for header in headers[1:]:
            new_label = TableLabelHeader(text=header, bold=True)
            table.add_widget(new_label)

        ## Add cells
        for row_ct in range(1, rows + 1):
            cells = (
                TableInput(text=f"P{row_ct}", size_hint_x=0.5),  # Process label
                TableInput(),  # Runtime input
                TableInput(),  # Time input
            )
            cells[0].input_filter = None
            cells[0].disabled = True

            for cell in cells:
                table.add_widget(cell)

    def _add_table_values(self):
        ## Access widget references
        container = self.ids["proc_table"]
        widgets = [widget for widget in reversed(container.children)]
        widgets = self.get_columns(widgets)

        ## Remove job column
        widgets = widgets[1:]

        ## Remove other headers
        widgets = list(zip(*widgets))
        widgets = widgets[1:]
        widgets = list(zip(*widgets))

        ## Test data
        data = core.LECTURE_EXAMPLES["Lecture_FCFS_SJF_1"]

        ## Insert test data
        for samples, widget_grp in zip(list(data.values())[1:], widgets):
            for sample, widget in zip(samples, widget_grp):
                widget.text = str(sample)

    ## TODO: Remove this
    @staticmethod
    def chunks(iterable, chunk_size):
        return [
            iterable[chunk_start : chunk_start + chunk_size]
            for chunk_start in range(
                #
                0,
                len(iterable) // chunk_size * chunk_size,
                chunk_size,
            )
        ]

    def get_columns(self, flat_grid, cols=3):
        """
        Customized chunking logic to separate by columns instead
        similar to effect to chunking+transposing
        """

        base_col_idx = range(0, len(flat_grid), cols)
        per_columns = [
            #
            [
                flat_grid[idx + offset]
                # flat_grid[idx + offset].text
                # idx + offset
                for idx in base_col_idx
            ]
            for offset in range(cols)
        ]

        return per_columns

    def validate_inputs(self, proc_container, scrmgr):
        ## Get relevant widgets list
        procs = [cell.text for cell in reversed(proc_container.children)]

        ## Extract data from widgets list
        procs = self.get_columns(procs)

        ## Perform validation
        try:
            procs = self._validate_jobs(procs)
        except AssertionError as error:
            app = MDApp.get_running_app()
            app.popup_message("warning", str(error))
            return

        ## Assign input data to app
        app = MDApp.get_running_app()
        app.inputs = procs

        ## Change screens
        scrmgr.current = "loading"

    def _validate_jobs(self, jobs):
        ## Condition: At least one (1) job must be entered
        ## Can(?) be inferred from a column, e.g. "Size"
        sizes = jobs[1][1:]
        fail_msg = "At least [b]one (1)[/b] job must be entered."
        assert (len(sizes) - sizes.count("")) >= 1, fail_msg

        ## "Transpose" data into table-like format
        table = list(zip(*jobs))

        ## Skip first row, i.e. column headers
        table_rows = iter(table)
        headers = next(table_rows)

        ## Initialize container for valid rows
        valids = []
        valids.append(headers)

        ## Iterate through data
        for row in table_rows:
            ## Extract row data
            job_name, *details = row

            ## Condition: Row must be complete
            empty_cell_ct = details.count("")
            if empty_cell_ct == 2:
                continue  # Skip empty rows

            fail_msg = f"Please complete data for [b]{job_name}[/b]."
            assert empty_cell_ct == 0, fail_msg

            ## Condition: No job row input must be "skipped"
            ...

            ## Add valid to "final" table
            valids.append(row)

        ## Revert transposition
        valids = list(zip(*valids))
        return valids


## Other screens


class LoadingScreen(Screen):
    dialog: MDDialog = None

    def _feedback_popup(self):
        if self.dialog is None:
            self.dialog = MDDialog(auto_dismiss=False)

        self.dialog.text = "Processing data. . ."
        self.dialog.open()

    def on_enter(self):
        # ## Show simple feedback popup
        self._feedback_popup()

        ## Put data processing in another thread so that UI will not lock up
        thread = threading.Thread(target=self._prepare_data)
        thread.start()

    def _generate_random_string(self, length=8):
        return "".join(random.sample(ascii_letters, k=length))

    def _prepare_data(self):
        ## Access input data
        app = MDApp.get_running_app()
        procs = app.inputs
        scheduling = app.scheduling

        ## Clean input data; transform into dictionaries
        procs = self._clean_jobs(procs)
        procs = pd.DataFrame(procs)

        ####################
        ## CORE PROCESSES ##
        ####################

        ## Pre-check inputs
        ## TODO: Remove this? Already checked at input screen...
        prelims = core.Preliminaries(procs.copy(), scheduling)
        procs_ready = prelims.processes

        ## Simulate processes
        sim = core.Simulation(procs_ready, scheduling)
        procs_props = sim.processes

        ## Calculate CPU utilization
        idle_time = procs_props["Idle"].sum()  # Sum of all idle periods
        duration = procs_props[
            "Finish"
        ].max()  # Time by which all processes are finished
        cpu_util = (1 - (idle_time / duration)) * 100

        ## Calculate Average Turnaround Time (ATA)
        ata = procs_props["Turnaround"].mean()

        ## Calculate Average Waiting Time (AWT)
        awt = procs_props["Waiting"].mean()

        ## Events list
        events = sim.events

        ## Event timepoints
        timepoints = sim.timepoints

        ## Gantt chart
        plot = core.Plotting(procs_props)
        byte_figures = plot.save_bytes_all()

        ## Assign property references to app
        app.characteristic_table = procs_props
        app.CPU_util = cpu_util
        app.ATA = ata
        app.AWT = awt
        app.events = events
        app.timepoints = timepoints
        app.plot = plot
        app.figures = byte_figures

        ## Part 2...
        self.dialog.text = "Creating screens; [i][b]this may take a while...[/i][/b]"
        Clock.schedule_once(self._prepare_data2, 0.25)
        # self._prepare_data2()
        # breakpoint()

    def _prepare_data2(self, *args):
        app = MDApp.get_running_app()
        root = app.root
        timepoints = app.timepoints

        ## Create screens
        self._clear_existing_outputs()

        base_screen_name = self._generate_random_string()
        app.current_screens_base = base_screen_name
        output_scrs = self._create_output_screens(base_name=base_screen_name, app=app)

        self._create_summary_screens(app=app)

        ## Proceed to first output screen
        self.dialog.dismiss()
        app.root.current = output_scrs[0]
        # app.root.current = "summary"

    def _clear_existing_outputs(self):
        app = MDApp.get_running_app()

        base = app.current_screens_base
        if base == "":
            return

        root = app.root
        for screen in root.screens[2:]:
            root.remove_widget(screen)

    @mainthread
    def _add_screen_byteimage(self, screen, byte_det):
        buf, extension = byte_det

        if buf.closed:
            print("???")
            return

        buf.seek(0)
        raw_image = CoreImage(buf, ext=extension)
        img = Image(texture=raw_image.texture)

        screen.ids["scatter_container"].add_widget(img)

    def _close_byte_buffers(self, figures):
        for buf, _ in figures:
            buf.close()

    def _create_table(self, container, data):
        ## Fill empty data
        if len(data.columns) == 1:
            data = data.drop(columns=data.columns)

        if len(data.columns) == 0:
            data["[i]No data available[/i]"] = ""

        ## Add headers
        headers = data.columns
        for header in headers:
            new_cell = TableLabelHeader(text=header, bold=True)
            container.add_widget(new_cell)

        ## Add data
        for row in data.itertuples():
            for value in row[1:]:
                new_cell = TableInput(text=str(value), disabled=True)
                container.add_widget(new_cell)

    def btn_func(self, root, new_scr, *args):
        root.current = new_scr

    def _create_output_screens(self, base_name, app):
        root = app.root
        timepoints = app.timepoints
        plot = app.plot
        figures = app.figures
        figure_idx = 0

        ## Add initial screen
        ...
        # timepoints = timepoints[1:]

        ## Create screens for each timepoint and customize details
        screen_names = []
        Init, Idling, Release, Arrival, _, *_ = core.EventType

        for idx, timepoint in enumerate(timepoints):
            ## Unpack timepoint props
            time, process_name, event_type, states = timepoint
            # print(timepoint)

            ## Create new screen
            ## CRITICAL: Mabagal gumawa ng new screen!
            screen_name = f"{base_name}_{idx}"
            new_screen = OutputScreen(name=screen_name)
            print(screen_name)

            ## Construct new header string
            if event_type == Init:
                header = "[i]Before [b]Time 0[/b]...[/i]"

            elif event_type == Idling:
                time = f"[i]Time {time}[/i]"
                event = f"[i][b]{str(event_type)}[/i][/b]"

                header = " • ".join([time, str(event)])

            else:
                time = f"[i]Time {time}[/i]"
                event = f"[b]{str(event_type)}[/b]"
                name = f"[font=RobotoLight]{process_name}[/font]"

                header = " • ".join([time, str(event), name])

            new_screen.ids["header_info"].text = header
            # print(header)

            ## Add byte image
            prev_pt = timepoints[max(idx - 1, 0)]
            _, _, prev_evtyp, _ = prev_pt

            # print(event_type, prev_evtyp)

            if (
                ## ...
                event_type == Release
                or ((event_type == Arrival) and (prev_evtyp == Idling))
                # or not (event_type == Init)
            ):
                figure_idx += 1

            # print(figure_idx, len(figures))
            byte_det = figures[figure_idx]
            self._add_screen_byteimage(new_screen, byte_det)

            ## Create tables
            self._create_table(new_screen.ids["waitings"], states)  # Process states

            ## Store screen details
            screen_names.append(screen_name)
            root.add_widget(new_screen)

        # ## Close all byte buffers
        # self._close_byte_buffers(figures)

        ## Disable Previous button in first screen
        scr_first = root.get_screen(screen_names[0])
        scr_first.ids["prev_button"].disabled = True

        ## Disable Next button in last screen
        scr_last = root.get_screen(screen_names[-1])
        scr_last.ids["next_button"].disabled = True

        ## Return list of screen names
        return screen_names

    def _create_events_list(self, container, data):
        ## Add headers
        headers = data.columns
        for header in headers:
            new_cell = TableLabelHeader(text=header, bold=True)
            container.add_widget(new_cell)

        ## Add data
        for row in data.itertuples():
            for value in row[1:]:
                new_cell = TableInput(text=str(value), disabled=True)
                container.add_widget(new_cell)

    # def _create_summary_screens(self, root, datas):
    def _create_summary_screens(self, app):
        summary = SummaryScreen(name="summary")

        char_tbl = app.characteristic_table
        events = app.events

        cpu = app.CPU_util
        ata = app.ATA
        awt = app.AWT
        cpu, ata, awt = [round(value, 2) for value in (app.CPU_util, app.ATA, app.AWT)]

        ## Rename columns to shorthand
        name_map = {
            "Burst Time": "Burst",
            "Arrival Time": "Arrival",
            "Turnaround": "Trnard.",
            "Waiting": "Wtng.",
        }
        char_tbl = char_tbl.rename(columns=name_map)

        ## Reduce columns
        # keep_cols = ["Arrival", "Burst", "Start", "Finish", "Trnard.", "Wtng."]
        keep_cols = ["Arrival", "Burst", "Start", "Finish"]
        char_tbl = char_tbl[keep_cols]

        ## Turn Process labels into column
        char_tbl = char_tbl.sort_index().reset_index()

        ## Create characteristics table
        self._create_table(summary.ids["char_table"], char_tbl)

        ## Create summary value table
        summ_vals = {
            "CPU Util.": [f"{cpu} %"],
            "ATA": [ata],
            "AWT": [awt],
        }
        summ_vals = pd.DataFrame(summ_vals)
        self._create_table(summary.ids["summary_table"], summ_vals)

        # ## Create summary value table
        # container = summary.ids["summary_table"]
        # for key, value in summ_struct.items():
        #     title = TableInput(text=key, bold=True, disabled=True)
        #     val = TableInput(text=str(value), disabled=True)

        #     for widget in (title, val):
        #         container.add_widget(widget)

        ## Create events table
        events = events.iloc[1:]
        self._create_table(summary.ids["events"], events)

        # breakpoint()

        # for wid, data in zip(("events", "job_table_summary", "summary_table"), datas):
        #     self._create_table(summary.ids[wid], data)

        app.root.add_widget(summary)

    def _clean_jobs(self, valid_jobs):  # TODO: Refactor this? Hahaha
        ## Get only data part | Remove headers
        data = list(zip(*valid_jobs))
        headers = data[0]
        data = data[1:]
        data = list(zip(*data))

        ## Turn inputs into actual numbers
        data[1] = [int(runtime) for runtime in data[1]]
        data[2] = [int(arrival) for arrival in data[2]]

        # ## Reattach headers
        # data = list(zip(*data))
        # data = [headers] + data
        # data = list(zip(*data))

        ## Transform into dictionary
        data = {header: d for header, d in zip(headers, data)}

        return data

    def _clean_partitions(self, partitions):  # TODO: Refactor this? Hahaha
        ## Get only data part | Remove headers
        data = list(zip(*partitions))
        data = data[1:]
        data = list(zip(*data))

        ## Turn sizes into actual numbers
        data[1] = [int(size) for size in data[1]]

        # ## Reattach headers
        # data = list(zip(*data))
        # data = [headers] + data
        # data = list(zip(*data))

        ## Map partition names to their sizes
        data = list(zip(*data))
        data = {name: size for name, size in data}

        return data


class ResizableDraggablePicture(ScatterLayout):
    """
    https://stackoverflow.com/questions/49807052/kivy-scroll-to-zoom
    """

    # scale: float = 3
    # scale_min: float = 1
    # scale_max: float
    base_factor: float = 1.5

    zoom_level_min = 0
    zoom_level_max = 3
    zoom_level = 0

    auto_bring_to_front = False
    do_rotation = False

    def on_touch_down(self, touch):
        if not touch.is_mouse_scrolling:
            super().on_touch_down(touch)
            return

        factor = None
        if touch.button == "scrolldown" and self.zoom_level < self.zoom_level_max:
            ## Zoom in
            factor = self.base_factor
            self.zoom_level += 1
        elif touch.button == "scrollup" and self.zoom_level > self.zoom_level_min:
            ## Zoom out
            factor = 1 / self.base_factor
            self.zoom_level -= 1

        if factor is not None:
            mat = Matrix().scale(factor, factor, factor)
            self.apply_transform(mat, anchor=touch.pos)

    """
    def on_touch_down(self, touch):
        # Override Scatter's `on_touch_down` behavior for mouse scroll
        if touch.is_mouse_scrolling:
            factor = None

            if touch.button == "scrolldown":
                # if self.scale < 10:
                #     self.scale = self.scale * 1.1
                if self.scale < self.scale_max:
                    factor = self.base_factor
            elif touch.button == "scrollup":
                # if self.scale > 1:
                #     self.scale = self.scale * 0.8
                if self.scale > self.scale_min:
                    factor = 1 / self.base_factor

            if factor is not None:
                mat = Matrix().scale(factor, factor, factor)
                self.apply_transform(mat, anchor=touch.pos)
        # If some other kind of "touch": Fall back on Scatter's behavior
        else:
            super(ResizableDraggablePicture, self).on_touch_down(touch)
    """


class OutputScreen(Screen):
    """
    byte_image_details: tuple

    @mainthread
    def on_pre_enter(self):
        buf, extension = self.byte_image_details
        raw_image = CoreImage(buf, ext=extension)

        img = Image(texture=raw_image.texture)
        self.ids["scatter_container"].add_widget(img)

        # self.ids["scatter_image"].texture = raw_image.texture
    """

    def btn_callback(self, type_):
        base, idx = self.name.split("_")
        idx = int(idx)

        if type_ == "prev":
            self.parent.current = f"{base}_{idx - 1}"

        elif type_ == "next":
            self.parent.current = f"{base}_{idx + 1}"


class SummaryScreen(Screen):
    ...
