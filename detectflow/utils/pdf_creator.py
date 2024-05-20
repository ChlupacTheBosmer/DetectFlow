try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    report_available = True
except ImportError:
    report_available = False

from PIL import Image as PILImage
from datetime import datetime, timedelta
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import io
import tempfile
import logging
import traceback

from detectflow.predict.results import DetectionBoxes

class PDFCreator():
    def __init__(self,
                 filename: str,
                 output_folder: str,
                 data: Optional[dict] = None
                 ):

        if not report_available:
            raise ImportError("PDFCreator requires the 'reportlab' package to be installed. Install the package with 'pip install detectflow[pdf]'.")

        # Set attributes
        self.filename = filename
        self.output_folder = output_folder
        self.filepath = os.path.join(output_folder, filename)
        self.data = data

    def init_doc(self):

        # Init pdf creation
        doc = SimpleDocTemplate(self.filepath, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        return doc, story, styles

    def build_doc(self, doc, story):
        success = False
        try:
            # Build PDF
            doc.build(story)
        except Exception as e:
            logging.error(f"ERROR: (PDF Creator) Error building PDF. Error: {e}")
        finally:
            if os.path.exists(self.filepath):
                logging.info(f"INFO: (PDF Creator) file <{self.filename}> created successfully.")
                success = True
        return success

    def create_pdf(self):
        '''Implement custom pdf format upon subclassing'''
        # Init the doc
        doc, story, styles = self.init_doc()

        # Prepare data
        if self.data is not None:
            data = self.data
        else:
            logging.error(f"ERROR: (PDF Creator) No data passed to the creator. Terminating...")
            return

        # Insert content

        # Build the doc
        self.build_doc(doc, story)

    def create_frame_image(self, frame, bboxes=None, fig_size=(2, 1.5)):
        img_data = None
        try:
            if frame is not None:
                fig, ax = plt.subplots(figsize=fig_size, dpi=100)
                ax.imshow(frame)
                if bboxes and isinstance(bboxes, DetectionBoxes):
                    for bbox in bboxes.xyxy:
                        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                                 linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                ax.axis('off')
                fig.tight_layout(pad=0)

                # Save the plot to a BytesIO object
                img_data = io.BytesIO()
                plt.savefig(img_data, format='png')
                img_data.seek(0)
        except Exception as e:
            logging.error(f"ERROR: (PDF Creator) Error drawing a plot from a frame. Error: {e}")
        finally:
            if fig:
                plt.close(fig)
        return img_data

    @staticmethod
    def create_plot_image(fig):
        img_data = None
        try:
            if fig is not None:
                img_data = io.BytesIO()
                fig.savefig(img_data, format='png', bbox_inches='tight')
                img_data.seek(0)
        except Exception as e:
            logging.error(f"ERROR: (PDF Creator) Error drawing a plot from a matlab plot. Error: {e}")
        finally:
            plt.close(fig)
        return img_data


class DiagPDFCreator(PDFCreator):
    def __init__(self,
                 filename: str,
                 output_folder: str,
                 data: Optional[dict] = None
                 ):

        super().__init__(filename, output_folder, data)

    def create_pdf(self):
        '''Implement custom pdf format upon subclassing'''
        # Init the doc
        doc, story, styles = self.init_doc()

        # Prepare data
        if self.data is None or not isinstance(self.data, dict):
            logging.error(
                f"ERROR: (PDF Creator) No data or invalid format passed to the creator. Data type: {type(self.data)}, expected: dict")
            return
        else:
            data = self.data
            try:
                basic_data = data.get("basic_data", {})
            except Exception as e:
                logging.error(f"ERROR: (PDF Creator) Error assigning basic data key. Error: {e}")

            try:
                # Fetch value from dictionary
                duration_delta = basic_data.get("duration", 0)

                # Format duration
                duration = duration_delta.total_seconds() if isinstance(duration_delta, timedelta) else 0
                duration_minutes = duration // 60
                duration_seconds = duration % 60
                duration_string = f"{int(duration_minutes):02d}:{int(duration_seconds):02d}"
            except Exception as e:
                logging.error(f"ERROR: (PDF Creator) Error assigning duration info. Error: {e}")
                duration_minutes = duration_seconds = duration_string = None

            # Init values
            try:
                # Fetch values from dictionary
                start_time = basic_data.get("start_time", 0)
                end_time = basic_data.get("end_time", 0)

                # Check if they are valid datetime objects
                if isinstance(start_time, datetime) and isinstance(end_time, datetime):
                    # Format the time as "HH:mm:ss"
                    start_time_string = start_time.strftime("%H:%M:%S")
                    end_time_string = end_time.strftime("%H:%M:%S")

                    # Format the date as "DD.MM.YYYY"
                    start_date_string = start_time.strftime("%d.%m.%Y")
                    end_date_string = end_time.strftime("%d.%m.%Y")
                else:
                    raise ValueError("start_time and end_time must be datetime objects")
            except Exception as e:
                logging.error(f"ERROR: (PDF Creator) Error assigning time info. Error: {e}")
                start_time_string = end_time_string = start_date_string = end_date_string = None

            # Prepare flower analysis data
            # Initialize variables to None
            try:
                # Fetch values from dictionary
                analysis_results = data.get("roi_data", {})

                # Start data
                number_of_flowers_start = analysis_results["Start"]["number_of_boxes"]
                max_area_start = analysis_results["Start"]["max_area"]
                min_area_start = analysis_results["Start"]["min_area"]
                area_range_start = analysis_results["Start"]["area_range"]
                area_variance_start = analysis_results["Start"]["area_variance"]
                discrepancies_start = analysis_results["Start"]["discrepancies_count"]

                # End data
                number_of_flowers_end = analysis_results["End"]["number_of_boxes"]
                max_area_end = analysis_results["End"]["max_area"]
                min_area_end = analysis_results["End"]["min_area"]
                area_range_end = analysis_results["End"]["area_range"]
                area_variance_end = analysis_results["End"]["area_variance"]
                discrepancies_end = analysis_results["End"]["discrepancies_count"]
            except KeyError as key:
                logging.error(
                    f"ERROR: (PDF Creator) KeyError: The key '{key}' was not found in the analysis_results dictionary.")
                number_of_flowers_start = max_area_start = min_area_start = area_range_start = area_variance_start = discrepancies_start = None
                number_of_flowers_end = max_area_end = min_area_end = area_range_end = area_variance_end = discrepancies_end = None

            # Motion data
            motion_data_entries = []
            high_movement_periods = []
            means = []
            motion_plots = []

            try:
                # Fetch data from dictionary
                motion_data = data.get("motion_data", {})

                if motion_data is not None:
                    if len(motion_data) > 0:
                        for key in motion_data:
                            means.append(motion_data[key].get('mean', 0))
                            motion_data_entries.append([f"Mean Motion ({key})", motion_data[key].get('mean', 'N/A')])
                            high_movement_periods.append(motion_data[key].get('high_movement_periods_t', [(0, 0)]))
                            motion_plots.append(motion_data[key].get('plot', None))
            except KeyError:
                logging.error(
                    f"ERROR: (PDF Creator) KeyError: The key '{key}' was not found in the motion_data dictionary.")

            # Calculate mean of means and determine windiness
            mean_motion = sum(means) / len(means) if len(means) > 0 and all(
                [mean is not None for mean in means]) else None
            wind = "Windy" if mean_motion is not None and mean_motion > 0.1 else "Calm"  # TODO: Make more sophisticated based on experience

            # Determine AI suitability
            # TODO: Create a metric for accessing the suitability for AI automated detection.
            # Consider accounting for, windiness, quality, blur, number of flowers and their size, confidence in flower detection

            # Heading and Subheading
            story.append(Paragraph("Diagnostic Report", styles['Title']))
            story.append(Paragraph(" ", styles['Title']))
            story.append(Paragraph(f'Basic Information', styles['Heading2']))

            # Define table data
            table_data = []
            try:
                table_data = [
                    ["Video ID", basic_data.get("video_id", "No ID")],
                    ["Recording ID", basic_data.get("recording_id", "No ID")],
                    ["Duration", duration_string],
                    ["Start Date", start_date_string],
                    ["Start Time", start_time_string],
                    ["End Date", end_date_string],
                    ["End Time", end_time_string],
                    ["Total Frames", basic_data.get("total_frames", "NA")],
                    ["Frame Rate", basic_data.get("frame_rate", "NA")],
                    ["Width", basic_data.get("frame_width", "NA")],
                    ["Height", basic_data.get("frame_height", "NA")],
                    ["Format", basic_data.get("format", "NA")],
                    ["Video Origin", basic_data.get("video_origin", "NA")],
                    ["Decord Validation", "decord" in basic_data.get("validated_methods", "NA")],
                    ["OpenCV Validation", "cv2" in basic_data.get("validated_methods", "NA")],
                    ["ImageIO Validation", "imageio" in basic_data.get("validated_methods", "NA")],
                    ["Daytime", data.get("daytime", "NA")],
                    ["Number of Flowers (Start)", number_of_flowers_start],
                    ["Number of Flowers (End)", number_of_flowers_end],
                    ["Mean Motion", mean_motion],
                    ["Wind", wind],
                    ["AI Suitability", "TBD"]
                ]
            except KeyError:
                logging.error(f"ERROR: (PDF Creator) KeyError: The key '{key}' was not found in the data dictionary.")
            except Exception as e:
                logging.error(f"ERROR: (PDF Creator) Error when formatting table data: {e}")
                traceback.print_exc()

            # Create and style the table
            table = Table(table_data, colWidths=[doc.width / 2.25] * 2)
            table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
            ]))
            story.append(table)

            # Example Frames Heading
            story.append(Paragraph("Example Frames", styles['Heading2']))

            # Draw example frames with bounding boxes
            frames = []
            ref_bboxes = []

            # Get data from dictionary
            try:
                frames = data.get("frames", [])
                ref_bboxes = data.get("roi_bboxes", [])
            except KeyError:
                logging.error(
                    f"ERROR: (PDF Creator) KeyError: The key '{key}' was not found in the motion_data dictionary.")

            frame_images = []
            bboxes = [ref_bboxes[0] if isinstance(ref_bboxes[0], DetectionBoxes) else None] * (len(frames) - 1) + \
                     [ref_bboxes[-1] if isinstance(ref_bboxes[-1], DetectionBoxes) else None]

            # Draw bounding boxes on the first and last frames
            for i, frame in enumerate(frames):

                # Create an image from a plot of a frame with bboxes
                img_data = self.create_frame_image(frame, bboxes[i])

                # Convert BytesIO to PIL Image
                if img_data is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                        pil_img = PILImage.open(img_data)
                        pil_img.save(tmpfile, format='PNG')
                        tmpfile_path = tmpfile.name

                    # Now tmpfile_path is the path to the saved image
                    frame_images.append(Image(tmpfile_path, width=2 * inch, height=1.5 * inch))

            # Assuming you have 12 frames for a 4x3 grid
            num_rows = 4
            num_cols = 3

            # Prepare table data
            table_data = []
            if frame_images:
                for i in range(num_rows):
                    if len(frame_images) >= (i + 1) * num_cols:
                        row_data = frame_images[i * num_cols:(i + 1) * num_cols]
                        table_data.append(row_data)

            # Create table
            frame_table = Table(table_data)
            frame_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),  # Optional grid style
            ]))

            story.append(frame_table)

            # Data about flower detections
            # First frame data
            start_data = [
                ["Number of Flowers", number_of_flowers_start],
                ["Max Area", max_area_start],
                ["Min Area", min_area_start],
                ["Area Range", area_range_start],
                ["Area Variance", area_variance_start],
                ["Discrepancies", discrepancies_start]
            ]

            # Last frame data
            end_data = [
                ["Number of Flowers", number_of_flowers_end],
                ["Max Area", max_area_end],
                ["Min Area", min_area_end],
                ["Area Range", area_range_end],
                ["Area Variance", area_variance_end],
                ["Discrepancies", discrepancies_end]
            ]

            # Add heading for first frame data
            story.append(Paragraph("First Frame Flower Data", styles['Heading2']))

            # Create and add table for first frame data
            start_table = Table(start_data, colWidths=[doc.width / 3.0] * 2)
            start_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
            ]))
            story.append(start_table)

            # Add heading for last frame data
            story.append(Paragraph("Last Frame Flower Data", styles['Heading2']))

            # Create and add table for last frame data
            end_table = Table(end_data, colWidths=[doc.width / 3.0] * 2)
            end_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
            ]))
            story.append(end_table)

            # Motion Detection Heading
            story.append(Paragraph("Motion Detection", styles['Heading2']))

            # Motion Detection Plots
            for plot, periods in zip(motion_plots, high_movement_periods):

                # Convert the plot to an image stream
                img_data = PDFCreator.create_plot_image(plot)

                # Convert BytesIO to PIL Image
                if img_data is not None:
                    pil_img = PILImage.open(img_data)

                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                        pil_img.save(tmpfile, format='PNG')
                        tmpfile_path = tmpfile.name

                    # Now tmpfile_path is the path to the saved image
                    story.append(Image(tmpfile_path, width=6 * inch, height=4.5 * inch))

                # Add high movement periods
                story.append(Paragraph("High Movement Periods", styles['Heading3']))
                if periods:
                    for period in periods:
                        story.append(Paragraph(self.print_time_in_minutes(period), styles['Bullet']))

            # Build the doc
            success = self.build_doc(doc, story)

            return success

    # Print the high movement periods in MM:ss format
    def print_time_in_minutes(self, period):
        start_minutes = int(period[0] // 60)
        start_seconds = int(period[0] % 60)
        end_minutes = int(period[1] // 60)
        end_seconds = int(period[1] % 60)
        return f"Start time: {start_minutes:02d}:{start_seconds:02d}, End time: {end_minutes:02d}:{end_seconds:02d}"

