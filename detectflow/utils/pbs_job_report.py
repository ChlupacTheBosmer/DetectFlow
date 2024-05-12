import os

class PBSJobReport:
    def __init__(self, job_data, logs):
        self.job_data = job_data
        self.logs = logs

    def analyze_exit_status(self):
        status = int(self.job_data['exit_status'])
        explanation = ""
        if status < 0:
            explanation = "The job was terminated by PBS, possibly due to exceeding a resource limit or other system-level issues."
        elif 0 <= status < 256:
            if status == 0:
                explanation = "The job completed successfully without any errors."
            else:
                explanation = f"The job exited with shell or process exit status {status}."
                common_exit_codes = {
                    1: "General errors or unspecified error.",
                    126: "Invoked command cannot execute.",
                    127: "Command not found.",
                    128: "Invalid argument to exit.",
                    130: "Script terminated by Control-C.",
                    137: "Process killed (SIGKILL).",
                    139: "Segmentation fault.",
                    143: "Termination (SIGTERM).",
                    255: "Exit status out of range."
                }
                explanation += f" Common cause: {common_exit_codes.get(status, 'Specific error undetermined')}"
        else:
            os_signal = status - 256
            explanation = f"The job was terminated by an OS signal (Signal number: {os_signal})."
            try:
                signal_name = os.strsignal(os_signal)
                explanation += f" Signal name: {signal_name}."
            except AttributeError:
                pass
        return explanation

    def generate_report_content(self):
        report_content = [
            ("Job ID", self.job_data['job_id']),
            ("Job Name", self.job_data['job_name']),
            ("Status", self.job_data['status']),
            ("Exit Status", self.job_data['exit_status']),
            ("Job Duration", f"Started: {self.job_data['start_time']}, Ended: {self.job_data['end_time']}"),
            ("Exit Status Analysis", self.analyze_exit_status())
        ]

        # Error logs
        if self.job_data['exit_status'] != '0':
            report_content.append(("Bash Error Log", self.logs.get('bash_error_log', 'N/A')))
            report_content.append(("Python Error Log", self.logs.get('python_error_log', 'N/A')))
        report_content.append(("Operation Log", self.logs.get('operation_log', 'No operation logs available.')))

        # Suggested actions
        actions = "No further action required." if self.job_data['exit_status'] == '0' else "Please review the error logs and correct any issues. Consider rerunning the job after making corrections."
        report_content.append(("Suggested Actions", actions))

        return report_content

    def generate_report(self, format='text'):
        content = self.generate_report_content()
        if format == 'text':
            return "\n\n".join(f"{key}: {value}" for key, value in content)
        elif format == 'html':
            html_content = "<table style='border: 1px solid black; border-collapse: collapse; width: 100%;'>"
            for key, value in content:
                html_content += f"<tr><th style='border: 1px solid black; background-color: #f2f2f2; padding: 8px;'>{key}</th><td style='border: 1px solid black; padding: 8px;'>{value}</td></tr>"
            html_content += "</table>"
            return html_content

