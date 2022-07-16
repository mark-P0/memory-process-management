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
        self._create_memory_layout()

        # ## DEBUG: Add sample values
        # self._add_table_values()
        # self._add_partition_values()

    def _create_table(self, rows=5):
        table = self.ids["job_table"]

        ## Add headers
        headers = ("Job No.", "Size", "Arrival Time", "Run Time (min.)")

        header = headers[0]
        new_label = TableLabelHeader(text=header, bold=True, size_hint_x=0.5)
        table.add_widget(new_label)

        for header in headers[1:]:
            new_label = TableLabelHeader(text=header, bold=True)
            table.add_widget(new_label)

        ## Add cells
        for row_ct in range(1, rows + 1):
            cells = (
                TableInput(text=f"Job {row_ct}", size_hint_x=0.5),  # Job label
                TableInput(),  # Size input
                TableInputTime(),  # Time input
                TableInput(),  # Runtime input
            )
            cells[0].input_filter = None
            cells[0].disabled = True

            for cell in cells:
                table.add_widget(cell)

    def _add_table_values(self):
        ## Access widget references
        container = self.ids["job_table"]
        widgets = [widget for widget in reversed(container.children)]
        widgets = self.get_columns(widgets)

        ## Remove job column
        widgets = widgets[1:]

        ## Remove other headers
        widgets = list(zip(*widgets))
        widgets = widgets[1:]
        widgets = list(zip(*widgets))

        ## Test data
        # fmt: off
        data = {
            # ## Lecture (Single cont.) | Example
            # "Job No.":          [   "Job 1",    "Job 2",    "Job 3"],
            # "Size":             [       100,        200,        300],
            # "Arrival Time":     [ "8:00 AM",  "8:15 AM",  "8:30 AM"],
            # "Run Time (min.)":  [        20,         30,         40],

            # ## Lecture (Single cont.) | Quiz-ish
            # "Job No.":          [   "Job 1",    "Job 2",    "Job 3",    "Job 4",     "Job 5"],
            # "Size":             [       150,        100,        250,        500,         400],
            # "Arrival Time":     [ "9:00 AM",  "9:15 AM",  "9:30 AM",  "9:45 AM",  "10:00 AM"],
            # "Run Time (min.)":  [        25,         20,         30,         45,          40],

            ## Exam 2 (Single cont.) | Midterm
            "Job No.":          [   "Job 1",    "Job 2",    "Job 3",    "Job 4",    "Job 5"],
            "Size":             [       120,        150,        500,        510,        100],
            "Arrival Time":     [ "8:00 AM",  "9:30 AM", "10:25 AM", "10:30 AM", "11:00 AM"],
            "Run Time (min.)":  [       100,        150,        170,        200,         70],

            # ## Lecture (Static)
            # "Job No.":          [   "Job 1",    "Job 2",    "Job 3"],
            # "Size":             [         5,         32,         50],
            # "Arrival Time":     [ "9:00 AM",  "9:05 AM",  "9:15 AM"],
            # "Run Time (min.)":  [        10,         20,         25],

            # ## Lecture (Dynamic) (First Fit)
            # "Job No.":          [   "Job 1",    "Job 2",    "Job 3"],
            # "Size":             [       200,        250,        300],
            # "Arrival Time":     [ "9:00 AM",  "9:10 AM",  "9:30 AM"],
            # "Run Time (min.)":  [        20,         25,         30],

            # ## Lecture (Dynamic) (Best Fit)
            # "Job No.":          [   "Job 1",    "Job 2",    "Job 3"],
            # "Size":             [       200,        250,        100],
            # "Arrival Time":     [ "9:00 AM",  "9:10 AM",  "9:30 AM"],
            # "Run Time (min.)":  [        20,         25,         30],

            # ## Lecture (Dynamic) (Modified) (Best for either First or Best Fit)
            # "Job No.":          [   "Job 1",    "Job 2",    "Job 3"],
            # "Size":             [       450,        250,        100],
            # "Arrival Time":     [ "9:00 AM",  "9:10 AM",  "9:30 AM"],
            # "Run Time (min.)":  [        20,         25,         30],

            # ## Exam 3 | Static | Dynamic, Best Fit
            # "Job No.":          [   "Job 1",    "Job 2",    "Job 3",    "Job 4",    "Job 5"],
            # "Size":             [        50,        510,        500,        200,         10],
            # "Arrival Time":     [ "9:00 AM", "10:00 AM", "10:30 AM", "12:00 PM",  "1:00 PM"],
            # "Run Time (min.)":  [        75,        180,        150,        100,         10],

        }
        # fmt: on

        ## Insert test data
        for samples, widget_grp in zip(list(data.values())[1:], widgets):
            for sample, widget in zip(samples, widget_grp):
                widget.text = str(sample)

    def _create_memory_layout(self, default_amount=5):
        layout = self.ids["memory_layout"]

        ## Add static partitions
        for pttn_ct in range(1, default_amount + 1):
            widgets = [
                TableInput(text=self.partition_name_template.format(pttn_ct)),
                TableInput(),
            ]
            widgets[0].input_filter = None
            widgets[0].disabled = True

            for widget in widgets:
                layout.add_widget(widget)

    def _add_partition_values(self):
        ## Access widget references
        container = self.ids["memory_layout"]
        widgets = [widget for widget in reversed(container.children)]
        widgets = self.get_columns(widgets, cols=2)

        ## Get "Size" widgets
        widgets = widgets[1][1:]
        # print(widgets)
        # return

        ## Test data
        # fmt: off

        os, memory = (
            # ## Lecture (Single cont.)
            # ## Lecture (Dynamic)
            # 32, 640

            ## Exam 2 (Single cont.) | Midterm
            312, 712

            # ## Lecture (Static)
            # 312, 1024

            # ## Exam 3 | Static | Dynamic, Best Fit
            # 312, 1212
        )

        data = {
            # ## Lecture (Single cont.)
            # # 'P1': 640 - 32
            # 'P1': memory - os

            ## Exam 2 (Single cont.) | Midterm
            # # 'P1': 712 - 312  # Exclude OS from given memory
            'P1': memory - os
            # 'P1': 712

            # ## Lecture (Static)
            # "P1": 8,
            # "P2": 32,
            # "P3": 32,
            # "P4": 120,
            # "P5": 520,

            # ## Lecture (Static) (Turned single cont.)
            # 'P1': 8 + 32 + 32 + 120 + 520

            # ## Exam 3 | Static
            # "P1": 50,
            # "P2": 250,
            # "P3": 600,

        }

        # fmt: on

        ## Insert test data
        for sample, widget in zip([os] + list(data.values()), widgets):
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

    def get_columns(self, flat_grid, cols=4):
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

    def add_partition(self, container):
        current_pttn_ct = (
            len(container.children) - 4  # Exclude headers and OS partition
        ) // 2  # Each partition are worth two cells
        new_pttn_label = self.partition_name_template.format(current_pttn_ct + 1)

        cells = [
            TableInput(text=new_pttn_label),
            TableInput(),
        ]
        for cell in cells:
            container.add_widget(cell)

    def remove_partition(self, container):
        current_pttn_ct = (
            len(container.children)
            - 4  # Exclude headers and OS partition
            - 2  # Exclude one static partition
        ) // 2  # Each partition are worth two cells

        if current_pttn_ct == 0:
            app = MDApp.get_running_app()
            message = "At least one (1) static partition is needed for [b]Single Contiguous[/b] mode"
            app.popup_message("warning", message)
            return

        for _ in range(2):
            container.remove_widget(container.children[0])

    def validate_inputs(self, job_container, memory_container, scrmgr):
        ## Get relevant widgets list
        # jobs = list(reversed(job_container.children))
        # pttns = list(reversed(memory_container.children))

        jobs = [cell.text for cell in reversed(job_container.children)]
        pttns = [cell.text for cell in reversed(memory_container.children)]

        ## Extract data from widgets list
        jobs = self.get_columns(jobs)
        pttns = self.get_columns(pttns, cols=2)

        ## Perform validation
        try:
            jobs = self._validate_jobs(jobs)
            pttns = self._validate_partitions(pttns)
        except AssertionError as error:
            app = MDApp.get_running_app()
            app.popup_message("warning", str(error))
            return

        ## Assign input data to app
        app = MDApp.get_running_app()
        app.jobs = jobs
        app.partitions = pttns

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
            details[1] = details[1][:-3]  # Discard period (` AM` | ` PM`)
            details[1] = details[1].replace(":", "")  # Remove time separator

            ## Condition: Row must be complete
            empty_cell_ct = details.count("")
            if empty_cell_ct == 3:
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

    def _validate_partitions(self, pttns):
        ## "Transpose" data into table-like format
        table = list(zip(*pttns))

        table_rows = iter(table)
        next(table_rows)  # Skip first row, i.e. column headers

        ## Condition: OS size must be provided
        os_label, os_size = next(table_rows)
        assert os_size != "", f"[b]{os_label} size[/b] cannot be empty."

        ## Condition: At least one (1) partition must be defined
        empty_pttn_ct = [row[1] for row in table[2:]].count("")
        fail_msg = "At least [b]one (1) partition size[/b] must be defined."
        assert empty_pttn_ct < len(table[2:]), fail_msg

        ## Condition: Partitions must be "continuous"
        pre_filter_enum = list(enumerate(table_rows))
        filtered = [row for row in pre_filter_enum if row[1][1] != ""]
        post_filter_enum = list(enumerate(filtered))

        valids = []
        for pttn in table[:2]:  # First two rows are immediately valid
            valids.append(pttn)

        for post, (pre, valid_pttn) in post_filter_enum:
            pttn_name = valid_pttn[0]
            fail_msg = f"Partition(s) [b]before {pttn_name}[/b] must have a size"
            assert post == pre, fail_msg

            valids.append(valid_pttn)

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
        job_table = app.jobs
        pttn_sizes = app.partitions

        ## Clean input data; transform into dictionaries
        job_table = self._clean_jobs(job_table)
        pttn_sizes = self._clean_partitions(pttn_sizes)

        ## Instantiate core process
        job_table = pd.DataFrame(job_table)
        memory = sum(pttn_sizes.values())
        os = pttn_sizes.pop("OS")

        coreproc = core.PartitionedStaticMEMMGMT(
            jobs=job_table,
            partitions=pttn_sizes,
            total_memory=memory,
            os_size=os,
        )

        ## Proceed through processes
        ## CRITICAL: Mabigat mag-plot
        # events = coreproc.events
        events, base_arrival = coreproc.events.copy(), coreproc.base_arrival
        events = coreproc._normalize_time_undo(events, ["Time"], base_arrival)

        summary_table = coreproc.generate_summary_table()
        timepoints, base_filename = coreproc.generate_timepoint_props(write_file=False)

        ## Assign property references to app
        app.job_table = job_table
        app.events = events
        app.summary_table = summary_table
        app.timepoints = timepoints
        app.memory_map_base_filename = base_filename

        ## Part 2...
        self.dialog.text = "Creating screens; [i][b]this may take a while...[/i][/b]"
        Clock.schedule_once(self._prepare_data2, 0.25)

    def _prepare_data2(self, *args):
        app = MDApp.get_running_app()

        ## Create screens
        self._clear_existing_outputs()

        base_screen_name = self._generate_random_string()
        app.current_screens_base = base_screen_name
        output_screens = self._create_output_screens(
            base_name=base_screen_name, timepoints=app.timepoints, root=app.root
        )

        # thread_target = lambda *args: self._create_output_screens(
        #     base_name=base_screen_name, timepoints=timepoints, root=app.root
        # )
        # thread = threading.Thread(target=thread_target)
        # thread.start()

        self._create_summary_screens(
            app.root, datas=(app.events, app.job_table, app.summary_table)
        )

        ## Proceed to first output screen
        app.root.current = output_screens[0]
        self.dialog.dismiss()

    def _clear_existing_outputs(self):
        app = MDApp.get_running_app()

        base = app.current_screens_base
        if base == "":
            return

        root = app.root
        for screen in root.screens[2:]:
            root.remove_widget(screen)

    @mainthread
    def _add_screen_byteimage(self, screen, byte_details):
        buf, extension = byte_details
        raw_image = CoreImage(buf, ext=extension)

        img = Image(texture=raw_image.texture)
        screen.ids["scatter_container"].add_widget(img)

        # screen.ids["scatter_image"].texture = raw_image.texture

        buf.close()

    def _create_table(self, container, data):
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

    def _create_output_screens(self, base_name, timepoints, root):
        ## Create screens for each timepoint
        ## and customize details
        screen_names = []

        for idx, timepoint in enumerate(timepoints):
            ## Unpack timepoint details
            time, name, event, waitings, pat, partition, figure, byte_det = timepoint

            ## Create new screen
            ## CRITICAL: Mabagal gumawa ng new screen!
            screen_name = f"{base_name}_{idx}"
            new_screen = OutputScreen(name=screen_name)
            print(screen_name)

            ## Construct new header string
            time = f"[b]{time}[/b]"
            name = f"[font=RobotoLight]{name}[/font]"
            event = f"[i]{str(event)}[/i]"
            partition = f"Partition: {str(partition)}"

            header = " | ".join([time, str(event), partition, name])
            new_screen.ids["header_info"].text = header

            ## Create image from bytes
            self._add_screen_byteimage(new_screen, byte_det)
            # new_screen.byte_image_details = byte_det

            ## Create tables
            waitings = waitings.T
            waitings = waitings.rename(columns=waitings.loc["Job No.", :])
            waitings = waitings.drop(index="Job No.")

            for wid, data in zip(("pat", "waitings"), (pat, waitings)):
                self._create_table(new_screen.ids[wid], data)

            ## Store screen details
            screen_names.append(screen_name)
            root.add_widget(new_screen)
            # self._close_figure(figure)

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

    def _create_summary_screens(self, root, datas):
        summary = SummaryScreen(name="summary")

        for wid, data in zip(("events", "job_table_summary", "summary_table"), datas):
            self._create_table(summary.ids[wid], data)

        root.add_widget(summary)

    def _clean_jobs(self, valid_jobs):  # TODO: Refactor this? Hahaha
        ## Get only data part | Remove headers
        data = list(zip(*valid_jobs))
        headers = data[0]
        data = data[1:]
        data = list(zip(*data))

        ## Turn sizes into actual numbers
        data[1] = [int(size) for size in data[1]]

        ## Turn runtimes into actual numbers
        data[3] = [int(runtime) for runtime in data[3]]

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
    ...

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
