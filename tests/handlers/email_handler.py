from detectflow.handlers.email_handler import EmailHandler
from detectflow.config.secret import EMAIL_PASSWORD, EMAIL_ADDRESS
import os
import unittest

class TestEmailHandler(unittest.TestCase):

    def setUp(self):
        email_password = os.getenv('EMAIL_PASSWORD')
        print(email_password)
        self.email_handler = EmailHandler("detectflow@gmail.com", EMAIL_PASSWORD)

    def test_format_email_with_image(self):
        short_text = "Resource monitoring plot"
        image_path = r"D:\Dílna\Kutění\Python\DetectFlow\tests\jobs\resource_usage_20240720_233919.png"
        body, image_attachment = self.email_handler.format_email_with_image(short_text, image_path)
        self.email_handler.send_email(EMAIL_ADDRESS, os.getenv('PBS_JOBID'), body, attachments={"plot.jpg": image_attachment})