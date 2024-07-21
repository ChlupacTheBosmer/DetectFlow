from detectflow.utils.profiling import ResourceMonitor, ResourceMonitorPID
import time
import unittest
import os


class TestResourceMonitor(unittest.TestCase):

    def test_resource_monitor(self):

        resource_monitor = ResourceMonitor(interval=1, plot_interval=20, show=True)
        resource_monitor.start()

        # Example of logging events
        try:
            for i in range(5):
                time.sleep(5)
                event_message = f"Event {i + 1}"
                color = 'blue' if i % 2 == 0 else 'green'
                resource_monitor.log_event(event_message, color)
                print(f"Logged: {event_message} with color {color}")

            # Simulate a long-running process
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping resource monitor.")
            resource_monitor.stop()

    def test_resource_monitor_with_email(self):

        from detectflow.handlers.email_handler import EmailHandler
        from detectflow.config.secret import EMAIL_PASSWORD, EMAIL_ADDRESS

        email_handler = EmailHandler("detectflow@gmail.com", EMAIL_PASSWORD)

        resource_monitor = ResourceMonitor(interval=1, plot_interval=20, show=False, email_handler=email_handler, email_address=EMAIL_ADDRESS, email_interval=5)
        resource_monitor.start()

        # Example of logging events
        try:
            for i in range(5):
                time.sleep(5)
                event_message = f"Event {i + 1}"
                color = 'blue' if i % 2 == 0 else 'green'
                resource_monitor.log_event(event_message, color)
                print(f"Logged: {event_message} with color {color}")

            # Simulate a long-running process
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping resource monitor.")
            resource_monitor.stop()

class TestResourceMonitorPID(unittest.TestCase):

    def test_resource_monitor(self):

        main_pid = os.getpid()
        resource_monitor = ResourceMonitorPID(main_pid=main_pid, interval=1, plot_interval=20, show=True)
        resource_monitor.start()

        # Example of logging events
        try:
            for i in range(5):
                time.sleep(5)
                event_message = f"Event {i + 1}"
                color = 'blue' if i % 2 == 0 else 'green'
                resource_monitor.log_event(event_message, color)
                print(f"Logged: {event_message} with color {color}")

            # Simulate a long-running process
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping resource monitor.")
            resource_monitor.stop()

    def test_resource_monitor_with_email(self):

        from detectflow.handlers.email_handler import EmailHandler
        from detectflow.config.secret import EMAIL_PASSWORD, EMAIL_ADDRESS

        email_handler = EmailHandler("detectflow@gmail.com", EMAIL_PASSWORD)

        main_pid = os.getpid()
        resource_monitor = ResourceMonitorPID(main_pid=main_pid, interval=1, plot_interval=20, show=True, email_handler=email_handler,
                                           email_address=EMAIL_ADDRESS, email_interval=5)
        resource_monitor.start()

        # Example of logging events
        try:
            for i in range(5):
                time.sleep(5)
                event_message = f"Event {i + 1}"
                color = 'blue' if i % 2 == 0 else 'green'
                resource_monitor.log_event(event_message, color)
                print(f"Logged: {event_message} with color {color}")

            # Simulate a long-running process
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping resource monitor.")
            resource_monitor.stop()


# Example usage
if __name__ == "__main__":
    unittest.main()

