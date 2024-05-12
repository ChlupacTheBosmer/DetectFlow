import re

class LogFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.bash_errors = []
        self.python_errors = []
        self.general_logs = []
        self.parse_log_file()

    def parse_log_file(self):
        """ Parse the log file to categorize entries into bash errors, python errors, and general logs. """
        python_traceback_active = False
        python_error_buffer = []

        try:
            with open(self.file_path, 'r') as file:
                for line in file:
                    if line.strip() == "":
                        continue

                    if 'Traceback' in line:
                        python_traceback_active = True
                        python_error_buffer = [line]
                    elif python_traceback_active:
                        if line.startswith((' ', '\t')) or 'File "' in line or 'Error' in line or line.startswith('Exception'):
                            python_error_buffer.append(line)
                        else:
                            self.python_errors.append(''.join(python_error_buffer))
                            python_traceback_active = False
                            python_error_buffer = []
                            self.process_line(line)
                            continue

                    if not python_traceback_active:
                        self.process_line(line)

            if python_error_buffer:
                self.python_errors.append(''.join(python_error_buffer))

        except FileNotFoundError:
            print(f"Error: The log file {self.file_path} does not exist.")
        except Exception as e:
            print(f"An error occurred while reading the log file: {e}")

    def process_line(self, line):
        """ Process a single line to classify it as a bash error or general log. """
        if re.search(r'command not found|syntax error|illegal command|no such file or directory', line, re.I):
            self.bash_errors.append(line)
        else:
            self.general_logs.append(line)

    def format_errors(self, errors):
        """ Return errors formatted with new lines. """
        return ''.join(errors)

    @property
    def formatted_bash_errors(self):
        """ Get formatted string of bash errors. """
        return self.format_errors(self.bash_errors)

    @property
    def formatted_python_errors(self):
        """ Get formatted string of python errors. """
        return self.format_errors(self.python_errors)

    @property
    def formatted_general_logs(self):
        """ Get formatted string of general log entries. """
        return self.format_errors(self.general_logs)
