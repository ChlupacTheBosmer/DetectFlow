from detectflow.utils.profiling import ResourceMonitor
import time
import unittest


class TestResourceMonitor(unittest.TestCase):

    def test_resource_monitor(self):

        resource_monitor = ResourceMonitor(interval=1, plot_interval=20, show=False)
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

