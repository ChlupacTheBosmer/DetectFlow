import time
from datetime import timedelta
import re
from functools import lru_cache
import os
import subprocess
import json
import logging
from typing import List, Tuple, Union, Optional, Dict

from detectflow.manipulators.manipulator import Manipulator
from detectflow.manipulators.s3_manipulator import S3Manipulator
from detectflow.manipulators.database_manipulator import DatabaseManipulator
from detectflow.handlers.config_handler import ConfigHandler
from detectflow.handlers.job_handler import JobHandler
from detectflow.handlers.ssh_handler import SSHHandler


class Scheduler(ConfigHandler):
    '''
    Note that for automatic remote job submission you ssh password should be assigned to environmental variable 'SSH_PASSWORD'
    '''

    def __init__(self,
                 output_path: str,
                 database_path='scheduler.db'):

        super().__init__(None, "json", {})

        self.output_path = output_path
        self.jobs_path = Manipulator.create_folders(["jobs"], output_path)[0]
        self.database_path = database_path
        self.database_manipulator = None

        # Assign attributes
        self.bucket_name = None
        self.directories = None
        self.python_script = None
        self.use_gpu = None
        self.resources = None
        self.job_config_path = None
        self.config_path = None
        self.job_config = None
        self.python_script = None
        self.username = None
        self.remote_host = None
        self.directory_filter = None

        # Init the DB
        self.setup_database()

    def setup_database(self):
        """Initialize or connect to a SQLite database to log job information."""

        # Create an instance of the DatabaseManipulator
        self.database_manipulator = DatabaseManipulator(self.database_path)

        # Create the table #TODO: Add task name
        columns = [
            ("job_id", "TEXT", "PRIMARY KEY"),
            ("job_name", "TEXT", ""),
            ("processed_s3_directory", "TEXT", ""),
            ("output_directory", "TEXT", ""),
            ("batch_script", "TEXT", ""),
            ("python_script", "TEXT", ""),
            ("config_data", "TEXT", ""),
            ("host_name", "TEXT", ""),
            ("requested_resources", "TEXT", ""),
            ("job_status", "TEXT", ""),
            ("exit_status", "TEXT", ""),
            ("result", "TEXT", "")
        ]
        self.database_manipulator.create_table("jobs", columns)

    def _get_directories(self):

        # Init the manipulator and validator
        s3_manipulator = S3Manipulator()

        # Validate bucket
        if not s3_manipulator.is_s3_bucket(self.bucket_name):
            raise ValueError(f"{self.bucket_name} is not a valid S3 bucket.")

        # Get the s3 paths to directories for the jobs
        directories = s3_manipulator.list_directories_s3(self.bucket_name, regex=self.directory_filter, full_path=True)

        # Filter directories based on the provided list of directory names
        directories = [d for d in directories if d.rstrip('/').split('/')[-1] in self.directories] if len(
            self.directories) > 0 else directories

        return directories

    def _prepare_ssh_auth(self, ssh_password):

        # Preapre SSH keys
        ssh_handler = SSHHandler()
        key_path = ssh_handler.generate_ssh_key(os.path.join(self.output_path, '.ssh', 'id_rsa'))
        if key_path is None:
            raise RuntimeError("SSH key generation failed. Authentication process terminated.")
        print(key_path)
        if not ssh_handler.copy_ssh_key_to_remote(key_path, self.username, self.remote_host, ssh_password):
            print("SSH key upload failed. Key might already be uploaded. Authentication might not work.")

    def submit_jobs(self,
                    bucket_name: str,
                    directory_filter: str = "",  # regex pattern to filter directories in the bucket
                    directories: Optional[Union[List, Tuple]] = [],
                    # or pass a list of directories directly (just dir names)
                    job_config: Optional[Dict] = None,  # Passing config as a dict will override the config file
                    job_config_path: Optional[str] = None,  # path can be passed and the cfg in the file with be used
                    python_script: Optional[str] = None,
                    use_gpu: bool = False,
                    resources: Dict = {},
                    username: Optional[str] = None,
                    remote_host: Optional[str] = None,
                    ssh_password: str = None,
                    ignore_ssh_auth: bool = False):
        """Submit a job for each directory in the S3 bucket."""

        # Assign attributes
        self.bucket_name = bucket_name
        self.directories = directories
        self.python_script = python_script
        self.use_gpu = use_gpu
        self.resources = resources
        self.job_config_path = job_config_path
        self.config_path = job_config_path
        self.job_config = self.load_config() if not job_config else job_config
        self.python_script = python_script
        self.username = username
        self.remote_host = remote_host

        if Manipulator.is_valid_regex(directory_filter):
            self.directory_filter = directory_filter
        else:
            raise ValueError(f"Invalid regex pattern: {directory_filter}")

        # Get directories based on user input
        s3_directories = self._get_directories()

        # Auth
        try:
            self._prepare_ssh_auth(ssh_password)
        except Exception as e:
            if not ignore_ssh_auth:
                raise RuntimeError("SHH authentication failed.") from e

        for s3_directory in s3_directories:
            job_name = f"{self.bucket_name}_{s3_directory.rstrip('/').split('/')[-1]}"
            job_output_path = Manipulator.create_folders([job_name], self.jobs_path)[0]
            resources = self.determine_resources()
            job_config_path = self.create_config_file(self._get_config_updates(s3_directory), job_name)
            job_script = self.generate_job_script(job_config_path, resources, job_name)
            job_id = self.submit_job(job_script)
            self.log_job(job_id, job_name, s3_directory, job_output_path, job_script, job_config_path, resources)

    def _format_duration(self, time_input):
        """
        Converts various time formats to a standardized 'hh:mm:ss' format representing duration.

        Parameters:
            time_input (str or int or float): Input time in various formats ('01:30', '1:15', '1', 1, 1.5, etc.)

        Returns:
            str: Time duration in 'hh:mm:ss' format.
        """
        try:
            if isinstance(time_input, (int, float)):  # Direct number input
                # Assuming the number is an hour count, convert to timedelta
                total_seconds = int(time_input * 3600)
            else:
                # Handle string input assuming it could be hours:minutes, hours:minutes:seconds or just hours
                parts = [int(part) for part in re.split("[:]", time_input)]
                if len(parts) == 1:
                    # Only hours are provided
                    total_seconds = parts[0] * 3600
                elif len(parts) == 2:
                    # Hours and minutes are provided
                    total_seconds = parts[0] * 3600 + parts[1] * 60
                elif len(parts) == 3:
                    # Hours, minutes, and seconds are provided
                    total_seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
                else:
                    return "Invalid time format"

            # Create a timedelta object from the total seconds
            td = timedelta(seconds=total_seconds)
            # Formatting to 'hh:mm:ss'
            return str(td)
        except ValueError:
            return "Invalid input"

    @lru_cache(maxsize=32)
    def determine_resources(self):
        """Placeholder for resource determination logic based on directory."""

        resources = {
            "walltime": self._format_duration(self.resources.get('walltime', 4)),
            "mem": f"{self.resources.get('mem', 64)}gb",
            "ncpus": self.resources.get('cpus', 10),
            "ngpus": 1 if self.use_gpu else 0,
            "scratch_local": f"{self.resources.get('scratch', 1)}gb",
            "queue": "gpu@pbs-m1.metacentrum.cz" if self.use_gpu else "default@pbs-m1.metacentrum.cz"
        }

        return resources

    def _get_config_updates(self, directory):

        # Prepare values for some config fields updates
        input_data = directory  # Input data - full path to single directory
        checkpoint_dir = self.output_path
        task_name = f"{self.bucket_name}_{directory.rstrip('/').split('/')[-1]}"

        # Configuration updates
        return {
            "input_data": input_data,
            "checkpoint_dir": checkpoint_dir,
            "task_name": task_name
        }

    def create_config_file(self, config_updates, config_file_name):
        """Generate a config file for the job."""

        # Get the template cfg
        task_config = self.job_config

        # Update config with values specific for this job
        if isinstance(config_updates, dict):
            for key, value in config_updates.items():
                if key in task_config:
                    task_config[key] = value

        # Save task specific cfg file
        try:
            task_config_path = os.path.join(self.jobs_path, config_file_name, f"{config_file_name}_config.json")
            with open(task_config_path, 'w') as file:
                json.dump(task_config, file, indent=4)
        except Exception as e:
            raise RuntimeError("Failed to save task-specific config file.") from e

        return task_config_path

    def generate_job_script(self, config_path, resources, task_name):
        """Generate the PBS job script."""

        job_script = f"""
                    #!/bin/bash
                    #PBS -N {task_name}
                    #PBS -q {resources['queue']}
                    #PBS -l walltime={resources['walltime']},select=1:ncpus={resources['ncpus']}:mem={resources['mem']}:ngpus={resources['ngpus'] if self.use_gpu else '0'}{':gpu_cap=sm_75' if self.use_gpu else ''}:scratch_local={resources['scratch_local']}

                    #PBS -o "{self.jobs_path}/{task_name}/{task_name}.out"
                    #PBS -e "{self.jobs_path}/{task_name}/{task_name}.err"

                    HOMEDIR="{self.output_path}"
                    JOBDIR="{self.output_path}/{task_name}/"
                    CONFIG="{config_path}"
                    SOURCE_FILE="{self.python_script}"
                    SING_IMAGE="/storage/brno2/home/hoidekr/Ultralytics/Ultralytics-8.0.199.sif"

                    # Check if the CONFIG variable is set and not empty
                    if [ -z "$CONFIG" ]; then
                        echo "Variable CONFIG is not set!" >&2
                        exit 1
                    fi

                    echo "Config is set to: $CONFIG"

                    # Append a line to a file "jobs_info.txt" containing the ID of the job, the 
                    # hostname of node it is run on and the path to a scratch directory. This 
                    # information helps to find a scratch directory in case the job fails and you 
                    # need to remove the scratch directory manually.

                    echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $HOMEDIR/jobs_info.txt

                    # Load modules here

                    # Test if scratch directory is set. If scratch directory is not set,
                    # issue an error message and exit.
                    test -n "$SCRATCHDIR" || {{
                        echo "Variable SCRATCHDIR is not set!" >&2
                        exit 1
                    }}

                    ################################################################################
                    # CALCULATIONS

                    #singularity exec -B $SCRATCHDIR:/mnt \
                    #$SING_IMAGE /bin/bash -c "python '$SOURCE_FILE' \
                                              #--config '$CONFIG'"

                    pwd > "$SCRATCHDIR"/test.txt
                    ################################################################################

                    # Copy everything from scratch directory to $JOBDIR
                    echo "Listing contents of scratch directory:"
                    ls -l $SCRATCHDIR

                    # Make sure that the output folder exists
                    mkdir -p "$JOBDIR"

                    if [ -d "$SCRATCHDIR" ] && [ "$(ls -A $SCRATCHDIR)" ]; then
                       echo "Copying files from scratch directory to job output directory."
                       cp -r "$SCRATCHDIR"/* "$JOBDIR" || echo "Failed to copy files."
                    else
                       echo "Scratch directory is empty or does not exist."
                    fi

                    clean_scratch
                    """
        job_script_path = os.path.join(self.jobs_path, task_name, f"{task_name}.sh")
        with open(job_script_path, 'w') as file:
            file.write(job_script)
        return job_script_path

    def submit_job(self, job_script):
        """Submit a job to the PBS queue and return the job ID."""
        print("Submitting job")
        ssh_command = ['ssh', 'USER@HOST', 'qsub', job_script]
        print("Executing command:", ' '.join(ssh_command))
        result = subprocess.run(ssh_command, capture_output=True, text=True)
        if result.returncode == 0:
            job_id = result.stdout.strip()
            print(f"Job {job_id} submitted successfully.")
            return job_id
        else:
            print(f"Failed to submit job: {result.stderr}")
            raise Exception(f"Failed to submit job: {result.stderr}")

    def log_job(self, job_id, job_name, s3_directory, job_output_path, job_script, config_path, resources):
        """Log job submission details to the SQLite database."""
        # Insert data into the table
        job_data = {
            "job_id": job_id,
            "job_name": job_name,
            "processed_s3_directory": s3_directory,
            "output_directory": job_output_path,
            "batch_script": job_script,
            "python_script": self.python_script,
            "config_data": config_path,
            "host_name": "PBS_HOST",
            "requested_resources": str(resources),
            "job_status": "SUBMITTED"
        }
        self.database_manipulator.insert("jobs", job_data, use_transaction=True)
        self.database_manipulator.close_connection()

    def monitor_jobs(self,
                     user_email: str,
                     s3_cfg_file: str = "/storage/brno2/home/USER/.s3.cfg",
                     llm_handler=None):
        """Monitor the status of submitted jobs, updating the database accordingly."""

        while True:

            # Print progress
            logging.info("Checking jobs...")

            # Get info about jobs
            query = "SELECT job_id, job_name, output_directory, batch_script, python_script, config_data, job_status FROM jobs WHERE result IS NULL"
            jobs = self.database_manipulator.fetch_all(query)

            # When no unprocessed jobs remain
            if len(jobs) == 0:
                self.database_manipulator.close_connection()
                logging.info("All jobs were processed.")
                break

            # For each job check status and handle accordingly
            for job in jobs:

                # Get job info from qstat command and append keys for script files
                info = self.fetch_job_info(job[0])
                info['batch_script'] = job[3]
                info['python_script'] = job[4]
                info['job_config'] = job[5]

                # If the status of the job changed
                status = info.get('status', 'UNKNOWN')
                exit_status = info.get('exit_status', None)
                result = None
                if status != job[6]:
                    if status == "FINISHED":
                        try:
                            # Initialize Job handler for the current job
                            job_handler = JobHandler(output_directory=job[2],
                                                     user_email=user_email,
                                                     s3_cfg_file=s3_cfg_file,
                                                     email_handler=None,
                                                     llm_handler=llm_handler)
                            # Handle job status change
                            job_handler.handle_finished_job(info)
                            result = "DONE OK" if exit_status == 0 else "ERROR OK"
                        except Exception as e:
                            logging.error(
                                f"Error when handling a finished job. Note that the backup of results and status check should be performed manually. Job ID: {job[0]}, Error: {e}")
                            result = "DONE ERROR" if exit_status == 0 else "ERROR ERROR"

                # Update job database
                update = {"job_status": status,
                          "result": result,
                          "exit_status": exit_status}
                self.database_manipulator.update("jobs", update, f"job_id = '{job[0]}'", use_transaction=True)
            self.database_manipulator.close_connection()

            # Print progress
            logging.info("Resting...")
            time.sleep(10)  # Check every once in a while

    def fetch_job_info(self, job_id):
        """Check the status of a job in the PBS system using qstat."""
        try:
            # Fetching detailed job information
            ssh_command = ['ssh', 'USER@HOST', 'qstat', '-f', job_id]
            result = subprocess.run(ssh_command, capture_output=True, text=True)
            if result.returncode != 0:
                # If job is not in the current list, try fetching finished job details
                ssh_command = ['ssh', 'USER@HOST', 'qstat', '-xf', job_id]
                result = subprocess.run(ssh_command, capture_output=True, text=True)
                if result.returncode != 0:
                    return {"status": "UNKNOWN", "message": "Job not found"}

            # Parse job details
            return self.parse_qstat_output(job_id, result.stdout)
        except Exception as e:
            print(f"Error checking job status: {e}")
            return {"status": "ERROR", "message": str(e)}

    #     def parse_qstat_output(self, job_id, output):
    #         """
    #         Parse the detailed output from qstat to extract job status and other information.

    #         Parameters:
    #             output (str): The raw string output from the 'qstat -f <job_id>' or 'qstat -xf <job_id>' command.

    #         Returns:
    #             dict: A dictionary containing parsed data from the qstat output. Example structure:
    #             {
    #                 'job_id': '1407015.pbs-m1.metacentrum.cz'
    #                 'job_name': 'cz1-m1_CZ1_M1_AraHir01',
    #                 'job_state': 'F',
    #                 'exit_status': '0',
    #                 'output_path': '/storage/brno2/home/USER/deploy/schedule/jobs/cz1-m1_CZ1_M1_AraHir01.out',
    #                 'error_path': '/storage/brno2/home/USER/deploy/schedule/jobs/cz1-m1_CZ1_M1_AraHir01.err',
    #                 'creation_time': 'Sat May 4 20:16:36 2024',
    #                 'queue_time': 'Sat May 4 20:16:36 2024',
    #                 'start_time': 'Sat May 4 20:16:54 2024',
    #                 'end_time': 'Sat May 4 20:17:22 2024',
    #                 'status': 'FINISHED'  # Translated from job_state
    #             }
    #         """
    #         job_info = {}
    #         job_info['job_id'] = job_id
    #         lines = output.split('\n')
    #         for line in lines:
    #             line = line.strip()
    #             if line.startswith('Job_Name ='):
    #                 job_info['job_name'] = line.split('=')[-1].strip()
    #             elif line.startswith('job_state ='):
    #                 job_info['job_state'] = line.split('=')[-1].strip()
    #             elif line.startswith('Exit_status='):
    #                 job_info['exit_status'] = int(line.split('=')[-1].strip())
    #             elif line.startswith('Output_Path ='):
    #                 job_info['output_path'] = line.split('=')[-1].strip().split(':')[1]  # Removing hostname if present
    #             elif line.startswith('Error_Path ='):
    #                 job_info['error_path'] = line.split('=')[-1].strip().split(':')[1]  # Removing hostname if present
    #             elif line.startswith('ctime ='):
    #                 job_info['creation_time'] = line.split('=')[-1].strip()
    #             elif line.startswith('qtime ='):
    #                 job_info['queue_time'] = line.split('=')[-1].strip()
    #             elif line.startswith('stime ='):
    #                 job_info['start_time'] = line.split('=')[-1].strip()
    #             elif line.startswith('obittime ='):
    #                 job_info['end_time'] = line.split('=')[-1].strip()

    #         # Convert job state to a more readable form
    #         state_map = {'Q': 'QUEUED', 'R': 'RUNNING', 'F': 'FINISHED', 'M': 'MOVED'}
    #         job_info['status'] = state_map.get(job_info.get('job_state', ''), 'UNKNOWN')

    #         return job_info

    def parse_qstat_output(self, job_id, output):
        """
        Parse the detailed output from qstat to extract job status and other information.

        Parameters:
            output (str): The raw string output from the 'qstat -f <job_id>' or 'qstat -xf <job_id>' command.

        Returns:
            dict: A dictionary containing parsed data from the qstat output. Example structure:
            {
                'job_id': '1407015.pbs-m1.metacentrum.cz'
                'job_name': 'cz1-m1_CZ1_M1_AraHir01',
                'job_state': 'F',
                'exit_status': '0',
                'output_path': '/storage/brno2/home/USER/deploy/schedule/jobs/cz1-m1_CZ1_M1_AraHir01.out',
                'error_path': '/storage/brno2/home/USER/deploy/schedule/jobs/cz1-m1_CZ1_M1_AraHir01.err',
                'creation_time': 'Sat May 4 20:16:36 2024',
                'queue_time': 'Sat May 4 20:16:36 2024',
                'start_time': 'Sat May 4 20:16:54 2024',
                'end_time': 'Sat May 4 20:17:22 2024',
                'status': 'FINISHED'  # Translated from job_state
            }
        """
        # Define keys of interest
        keys_of_interest = {'Job_Name': 'job_name',
                            'job_state': 'job_state',
                            'Exit_status': 'exit_status',
                            'Output_Path': 'output_path',
                            'Error_Path': 'error_path',
                            'ctime': 'creation_time',
                            'qtime': 'queue_time',
                            'stime': 'start_time',
                            'obittime': 'end_time'
                            }

        job_info = {'job_id': job_id}
        current_key = None
        buffer = ""

        for line in output.split('\n'):
            stripped_line = line.strip()
            if '=' in stripped_line:

                # Determine if this is a new key
                key, value = stripped_line.split('=', 1)
                key = key.strip()

                # Save the previous key-value pair if there was an ongoing key
                if current_key:
                    job_info[keys_of_interest.get(current_key)] = buffer.strip()

                if key in keys_of_interest:

                    # Start a new key
                    current_key = key
                    buffer = value.strip()
                else:
                    current_key = None  # Ignore lines from keys that are not of interest
            elif current_key:
                # This is a continuation of the previous key
                buffer += '' + stripped_line  # Append with a space to handle broken lines correctly

        # Catch any remaining data that hasn't been saved yet
        if current_key and buffer:
            job_info[keys_of_interest.get(current_key)] = buffer.strip()

        # Post-process specific fields
        if 'output_path' in job_info:
            job_info['output_path'] = job_info['output_path'].split(':', 1)[1]
        if 'error_path' in job_info:
            job_info['error_path'] = job_info['error_path'].split(':', 1)[1]
        if 'exit_status' in job_info:
            job_info['exit_status'] = int(job_info['exit_status'])

        # Map job state to a more readable form
        state_map = {'Q': 'QUEUED', 'R': 'RUNNING', 'F': 'FINISHED', 'M': 'MOVED'}
        job_info['status'] = state_map.get(job_info.get('job_state', 'UNKNOWN'), 'UNKNOWN')

        return job_info

    def validate_config(self):
        """
        Valdiate config is not required as it dynamically changes but method must be implemented.
        """
        required_keys = {}  # No required keys
        pass
