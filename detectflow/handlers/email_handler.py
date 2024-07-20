import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
import re
import os


class EmailHandler:
    def __init__(self, sender_email, app_password: str = "AUTH_INFO"):
        self.sender_email = sender_email
        self.app_password = app_password
        self.smtp_server = "smtp.gmail.com"
        self.port = 587  # Port for starttls

        self.signature = """
                            <html>
                                <body>
                                    <hr>
                                    <p>DetectFlow Metacentrum Bot</p>
                                    <img src="https://raw.githubusercontent.com/ChlupacTheBosmer/static_resources/main/detectflow_small.png" alt="Logo" width="150" height="150">
                                    <p>This email was automatically sent from the DetectFlow deployment bot running on the Metacentrum cluster.<br>
                                    Please do not respond. In case of an error please contact Petr Chlup via GitHub.</p>
                                </body>
                            </html>
                        """

    @staticmethod
    def format_data_for_email(data):
        """
        Format nested dictionary data into a string suitable for email content.

        Args:
        data (dict): A dictionary where keys are headers and values are dictionaries of data.

        Returns:
        str: A formatted string with headers and key-value pairs from nested dictionaries.
        """
        formatted_string = ""
        for header, details in data.items():
            # Add header
            formatted_string += f"{header}:\n"
            # Add each key-value pair under the header
            if isinstance(details, dict):
                for key, value in details.items():
                    formatted_string += f"  {key}: {value}\n"
            # Add a newline after each section for better readability
            formatted_string += "\n"

        return formatted_string.strip()  # Remove the last newline to tidy up the output

    @staticmethod
    def format_data_for_email_as_table(data):
        """
        Format nested dictionary data into an HTML table suitable for email content.

        Args:
        data (dict): A dictionary where keys are headers and values are dictionaries of data.

        Returns:
        str: A formatted HTML string with headers and key-value pairs from nested dictionaries as a table.
        """
        # Start the HTML table
        html_table = "<table style='border-collapse: collapse; width: 100%;'>"

        for header, details in data.items():
            # Add a header row for each main dictionary entry
            html_table += f"<tr><th colspan='2' style='border: 1px solid black; background-color: #f2f2f2; padding: 8px; text-align: left;'>{header}</th></tr>"

            # Add data rows
            if isinstance(details, dict):
                for key, value in details.items():
                    html_table += f"<tr><td style='border: 1px solid black; padding: 8px;'>{key}</td><td style='border: 1px solid black; padding: 8px;'>{value}</td></tr>"

        # Close the table
        html_table += "</table>"

        return html_table

    def format_email_with_image(self, short_text, image_path):
        if not os.path.isfile(image_path):
            raise ValueError("The provided image path does not exist.")

        # Create the HTML body with the table and image
        body = f"""
        <html>
            <body>
                <table style="border: 1px solid black; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 10px;">{short_text}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px;">
                            <img src="cid:image1" style="max-width: 100%; height: auto;" />
                        </td>
                    </tr>
                </table>
            </body>
        </html>
        """

        # Prepare the image attachment
        with open(image_path, 'rb') as img:
            mime = MIMEImage(img.read())
            mime.add_header('Content-ID', '<image1>')
            mime.add_header('Content-Disposition', 'inline', filename=os.path.basename(image_path))

        return body, mime

    def process_email_text(self, email_text):
        '''
        Method that will process string and remove a subject line and any placeholders to cleanup an automatically generaed email string.
        '''

        # Split the text into lines
        lines = email_text.split('\n')

        subject = None
        new_lines = []
        found_subject = False

        # Iterate through each line
        for line in lines:
            # Check for the subject line
            if re.match(r'^Subject *[:-]', line, re.IGNORECASE):
                # Extract the subject after "Subject:" or "Subject -" etc.
                subject = re.split(r'^Subject *[:-] *', line, flags=re.IGNORECASE)[1]
                found_subject = True
                continue  # Skip adding this line to new_lines

            # Skip empty lines immediately following the subject line
            if found_subject and line.strip() == '':
                continue

            # Reset found_subject flag if non-empty line is encountered after subject
            if line.strip() != '':
                found_subject = False

            # Add line if it's not part of the removed sections
            if not found_subject:
                new_lines.append(line)

        # Remove placeholder fields in square brackets
        new_text = '\n'.join(new_lines)
        new_text = re.sub(r'\[[^\]]*\]', '', new_text)

        return subject, new_text

    @staticmethod
    def is_html(s):
        """ Check if the string contains HTML tags. """
        return bool(re.search(r'<[a-z][\s\S]*>', s, re.IGNORECASE))

    def send_email(self, receiver_email, subject, body, appendix=None, attachments=None):
        """
        Send an email using the initialized credentials and server details.

        Args:
        receiver_email (str): Email address of the receiver.
        subject (str): Subject of the email.
        body (str): Plain text message body of the email.
        appendix (str): Optional appendix text to add below the main body.
        attachments (dict): Optional dictionary of filenames and their file paths to be attached.
        """
        from detectflow.validators.validator import Validator

        # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = receiver_email
        message["Subject"] = subject

        # Add body to email
        if body and EmailHandler.is_html(body):
            message.attach(MIMEText(body, "html"))
        elif body:
            message.attach(MIMEText(body, "plain"))

        # Add appendix if provided and determine if appendix contains HTML
        if appendix and EmailHandler.is_html(appendix):
            appendix_text = f"<p>Appendix:</p>{appendix}"
            message.attach(MIMEText(appendix_text, "html"))
        elif appendix:
            appendix_text = f"\nAppendix:\n{appendix}"
            message.attach(MIMEText(appendix_text, "plain"))

        # Attach files if provided
        if attachments:
            for filename, filepath in attachments.items():
                if isinstance(filepath, MIMEImage):
                    message.attach(filepath)
                    continue
                # Open file in binary mode
                if Validator.is_valid_file_path(filepath):
                    with open(filepath, "rb") as attachment:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(attachment.read())

                    # Encode file in ASCII characters to send by email
                    encoders.encode_base64(part)

                    # Add header as key/value pair to attachment part
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {filename}",
                    )

                    # Attach the file to the message
                    message.attach(part)

        # Attach HTML signature
        message.attach(MIMEText(self.signature, 'html'))

        try:
            # Connect to SMTP server and send email
            with smtplib.SMTP(self.smtp_server, self.port) as server:
                server.starttls()  # Secure the connection
                server.login(self.sender_email, self.app_password)
                server.sendmail(self.sender_email, receiver_email, message.as_string())
            print("Email sent successfully!")
        except Exception as e:
            print(f"An error occurred: {e}")

