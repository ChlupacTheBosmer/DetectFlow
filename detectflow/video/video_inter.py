# This file contains the video data classes
#
# Default python modules
import datetime
import os
import time
from datetime import datetime
import cv2
import logging

# Other modules
import pandas as pd
from hachoir.metadata import extractMetadata
from hachoir.parser import createParser

# Modules of DetectFlow
from video_passive import VideoFilePassive


class VideoFileInteractive(VideoFilePassive):
    __slots__ = ('end_time', 'main_window', 'manual_text_input_window', 'ocr_roi', 'start_time', 'sec_OCR')

    def __init__(self, filepath, root=None, ocr_roi: tuple = (0, 0, 500, 60),
                 initiate_start_and_end_times: bool = True):

        # Superclass init func call
        super().__init__(filepath)

        # Define variables
        self.manual_text_input_window = None
        self.main_window = root
        self.ocr_roi = ocr_roi

        # If relevant initiate the times
        if initiate_start_and_end_times:
            self.start_time, self.end_time = self.get_video_start_end_times()

    def get_video_start_end_times(self):

        # Define logger
        logging.debug(f"Running function get_video_start_end_times({self.filepath})")
        video_filename = os.path.basename(self.filepath)
        logging.debug(' '.join(["Processing video file -", self.filepath]))

        # Get start time from metadata but because the metadata often contain wrong hour number, will only use seconds
        start_time_meta, success = self.get_time_from_video_metadata(self.filepath, "start")

        # If failed to get time from metadata obtain it manually
        if not success:
            import vision_AI
            # Get the time in seconds manually
            # frame = self.get_video_frame("start")
            frame = self.read_video_frame(24, False)[0][3]

            success, extracted_time = vision_AI.get_text_with_OCR(frame)
            if not success:
                start_time_seconds, success = self.get_time_manually(frame)
            else:
                start_time_seconds = str(extracted_time.second)

        else:
            start_time_seconds = start_time_meta[-2:]

        # Now get the date, hour and minute from the filename
        filename_parts = video_filename[:-4].split("_")
        start_time_minutes = "_".join(
            [filename_parts[len(filename_parts) - 3], filename_parts[len(filename_parts) - 2],
             filename_parts[len(filename_parts) - 1]])  # creates timestamp

        # Construct the string
        start_time_str = '_'.join([start_time_minutes, start_time_seconds])
        start_time = pd.to_datetime(start_time_str, format='%Y%m%d_%H_%M_%S')
        self.start_time = start_time

        # Get end time
        end_time_meta, success = self.get_time_from_video_metadata(self.filepath, "end")

        # If failed to get time from metadata obtain it manually
        if not success:
            import vision_AI
            # Get the time in seconds manually
            # frame = self.get_video_frame("end")
            frame = self.read_video_frame(self.total_frames - 10, False)[0][3]

            success, extracted_time = vision_AI.get_text_with_OCR(frame)
            if not success:
                end_time_seconds, success = self.get_time_manually(frame)
            else:
                end_time_seconds = str(extracted_time.second)

            # 15 minute duration of the video is assumed and the manually extracted seconds are added
            delta = 15 + (int(end_time_seconds) // 60)
            end_time_seconds = str(int(end_time_seconds) % 60)

            # Construct the string
            end_time_str = '_'.join([start_time_minutes, end_time_seconds])
            end_time = pd.to_datetime(end_time_str, format='%Y%m%d_%H_%M_%S')
            end_time = end_time + pd.Timedelta(minutes=int(delta))
        else:
            end_time_str = '_'.join(
                [filename_parts[len(filename_parts) - 3], filename_parts[len(filename_parts) - 2], end_time_meta[-5:]])
            end_time = pd.to_datetime(end_time_str, format='%Y%m%d_%H_%M_%S')
        # print(f"start: {start_time}. end: {end_time}")
        self.end_time = end_time
        return start_time, end_time

    def get_time_from_video_metadata(self, video_filepath, start_or_end):
        return_time = None
        success = False
        if start_or_end == "start":
            try:
                # Get the creation date from metadata
                parser = createParser(video_filepath)
                metadata = extractMetadata(parser)
                modify_date = str(metadata.get("creation_date"))  # 2022-05-24 08:29:09

                # Convert it into the correct string format
                original_datetime = datetime.strptime(modify_date, '%Y-%m-%d %H:%M:%S')
                return_time = original_datetime.strftime('%Y%m%d_%H_%M_%S')
                logging.debug("Obtained video start time from metadata.")
                success = True
            except Exception as e:
                success = False
        elif start_or_end == "end":
            try:
                # Get the creation date and duration
                modify_date = self.start_time
                duration = self.get_video_duration()

                # Calculate the end time by adding the duration to the creation time
                end_time = modify_date + duration

                # Convert end time to the desired format
                return_time = end_time.strftime('%Y%m%d_%H_%M_%S')
                logging.debug("Obtained video end time from metadata.")
                success = True
            except:
                success = False
        return return_time, success

    def get_time_manually(self, frame):

        import tkinter as tk
        from PIL import Image, ImageTk

        def submit_time(manual_input_value):
            logging.debug(f'Running function submit_time({manual_input_value})')

            # Define variables
            text = manual_input_value.get()
            dialog_window = self.manual_text_input_window

            # Validate text
            if not text.isdigit() or len(text) != 2 or int(text) > 59:
                # execute code here for when text is not in "SS" format
                logging.warning(
                    "Manual input is not in the correct format. The value will be set to an arbitrary 00.")
                text = '00'
            else:
                logging.debug("Video times were not extracted from metadata. Resolved manually.")
            self.sec_OCR = text
            while True:
                try:
                    if dialog_window.winfo_exists():
                        dialog_window.quit()
                        dialog_window.destroy()
                        break
                except:
                    time.sleep(0.1)
                    break

        # Define variables
        root = self.main_window

        # Loop until the main window is created
        while True:
            try:
                if root.winfo_exists():
                    break
            except:
                time.sleep(0.1)

        # Create the dialog window
        dialog = tk.Toplevel(root)
        self.manual_text_input_window = dialog
        try:
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
        except:
            screen_width = 1920
            screen_height = 1080

        dialog.wm_attributes("-topmost", 1)
        dialog.title("Time Input")

        # convert frame to tkinter image
        text_roi = self.ocr_roi
        x, y, w, h = text_roi
        img_frame_width = min(screen_width // 2, w * 2)
        img_frame_height = min(screen_height // 2, h * 2)
        frame = frame[y:y + h, x:x + w]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_frame_width, img_frame_height), Image.LANCZOS)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        img_width = img.width()
        img_height = img.height()

        # Add image frame containing the video frame
        img_frame = tk.Frame(dialog, width=img_width, height=img_height)
        img_frame.pack(side=tk.TOP, pady=(0, 0))
        img_frame.pack_propagate(False)

        # Add image to image frame
        img_label = tk.Label(img_frame, image=img)
        img_label.pack(side=tk.TOP, pady=(0, 0))

        # Get the name of the video time to display it in the label.
        # The user then can decide whether ti is ok and eventually try to change it.
        filename_parts = os.path.basename(self.filepath)[:-4].split("_")
        start_time_minutes = ":".join(
            [filename_parts[len(filename_parts) - 2],
             filename_parts[len(filename_parts) - 1]])

        # Add label
        text_field = tk.Text(dialog, height=4, width=120, font=("Arial", 10))
        text_field.insert(tk.END,
                          "The OCR detection apparently failed."
                          "\nEnter the last two digits of the security camera watermark (number of seconds)."
                          "\nThis will ensure cropping will happen at the right times."
                          f"\nStart of the video from filename: {start_time_minutes}")
        text_field.configure(state="disabled", highlightthickness=1, relief="flat", background=dialog.cget('bg'))
        text_field.tag_configure("center", justify="center")
        text_field.tag_add("center", "1.0", "end")
        text_field.pack(side=tk.TOP, padx=(0, 0))
        label = tk.Label(dialog, text="Enter text", font=("Arial", 10), background=dialog.cget('bg'))
        label.pack(pady=2)

        # Add input field
        j = 0
        input_field = tk.Entry(dialog, font=("Arial", 10), width=4)
        input_field.pack(pady=2)
        input_field.bind("<Return>", lambda k=j: submit_time(input_field))
        input_field.focus()

        # Add submit button
        submit_button = tk.Button(dialog, text="Submit", font=("Arial", 10),
                                  command=lambda k=j: submit_time(input_field))
        submit_button.pack(pady=2)

        # Position the window
        dialog_width = dialog.winfo_reqwidth() + img_frame_width
        dialog_height = dialog.winfo_reqheight() + img_frame_height
        dialog_pos_x = int((screen_width // 2) - (img_frame_width // 2))
        dialog_pos_y = 0
        dialog.geometry(f"{dialog_width}x{dialog_height}+{dialog_pos_x}+{dialog_pos_y}")

        # Start the dialog
        dialog.mainloop()

        # When dialog is closed
        return_time = self.sec_OCR
        if len(return_time) > 0:
            success = True
        else:
            success = False
        return return_time, success
