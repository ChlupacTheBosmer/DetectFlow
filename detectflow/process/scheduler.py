import time
from datetime import timedelta
import re
from functools import lru_cache
import os
import subprocess
import json
import logging
from typing import List, Tuple, Union, Optional, Dict
import paramiko

from detectflow.manipulators.manipulator import Manipulator
from detectflow.manipulators.s3_manipulator import S3Manipulator
from detectflow.manipulators.database_manipulator import DatabaseManipulator
from detectflow.handlers.config_handler import ConfigHandler
from detectflow.handlers.job_handler import JobHandler
from detectflow.handlers.ssh_handler import SSHHandler
from detectflow.config import S3_CONFIG
from detectflow.utils import MACOS, LINUX, WINDOWS
from detectflow.utils.input import format_duration
from detectflow.utils.jobs import get_job_info


class Scheduler(ConfigHandler):
    '''
    Note that for automatic remote job submission you ssh password should be assigned to environmental variable 'SSH_PASSWORD'

    Note that if running on a local machine care must be taken when setting the arguments:
    - local_work_dir: path to the local working directory
    - remote_work_dir: path to the remote working directory
    - database_path: path to the database file, which should be stored locally
    - python_script_path: REMOTE path to the python script to be executed
    - config_path: LOCAL path to the job configuration file

    '''

    def __init__(self, local_work_dir: str, remote_work_dir: Optional[str] = None, database_path='scheduler.db'):

        super().__init__(None, "json", {})

        self.local_work_dir = local_work_dir
        self.remote_work_dir = remote_work_dir.replace("\\", "/")
        self.jobs_dir = Manipulator.create_folders(["jobs"], local_work_dir)[0]
        self.database_path = database_path
        self.database_manipulator = None
        self._is_remote = True if WINDOWS or MACOS else False if LINUX and remote_work_dir else True

        # Assign attributes
        self.bucket_name = None
        self.directories = None
        self.use_gpu = None
        self.resources = None
        self.config_path = None
        self.config = None
        self.python_script_path = None
        self.username = None
        self.remote_host = None
        self.directory_filter = None

        # Init the DB
        self._setup_database()

    def _validate_config(self):
        """
        Validate config is not required as it dynamically changes but method must be implemented.
        """
        required_keys = {}  # No required keys
        pass

    def _setup_database(self):
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

    @lru_cache(maxsize=32)
    def _determine_resources(self):
        """Placeholder for resource determination logic based on directory."""

        resources = {
            "walltime": format_duration(self.resources.get('walltime', 4)),
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
        checkpoint_dir = self.local_work_dir if not self._is_remote else self.remote_work_dir
        task_name = f"{self.bucket_name}_{directory.rstrip('/').split('/')[-1]}"

        # Configuration updates
        return {
            "input_data": input_data,
            "checkpoint_dir": checkpoint_dir,
            "task_name": task_name
        }

    def _generate_config_file(self, config_file_name, config_updates):
        """Generate a config file for the job."""

        # Get the template cfg
        task_config = self.config

        # Update config with values specific for this job
        if isinstance(config_updates, dict):
            for key, value in config_updates.items():
                if key in task_config:
                    task_config[key] = value

        # Save task specific cfg file
        try:
            task_config_path = os.path.join(self.jobs_dir, config_file_name, f"{config_file_name}_config.json")
            with open(task_config_path, 'w') as file:
                json.dump(task_config, file, indent=4)
        except Exception as e:
            raise RuntimeError("Failed to save task-specific config file.") from e

        return task_config_path

    def _generate_job_script(self, job_name, job_resources, python_script_path, config_path):
        """Generate the PBS job script.
        :param python_script_path:
        """

        if self._is_remote:
            home_dir = self.remote_work_dir
            task_folder = "/".join((self.remote_work_dir, 'jobs', job_name)).replace("\\", "/")
            config_path = "/".join((task_folder, os.path.basename(config_path))).replace("\\", "/")
            python_script_path = "/".join((task_folder, os.path.basename(python_script_path))).replace("\\", "/")
        else:
            home_dir = self.local_work_dir
            task_folder = os.path.join(self.jobs_dir, job_name)
            config_path = config_path
            python_script_path = python_script_path

        job_script = f"""#!/bin/bash
        #PBS -N {job_name}
        #PBS -q {job_resources['queue']}
        #PBS -l walltime={job_resources['walltime']},select=1:ncpus={job_resources['ncpus']}:mem={job_resources['mem']}:ngpus={job_resources['ngpus'] if self.use_gpu else '0'}{':gpu_cap=sm_75' if self.use_gpu else ''}:scratch_local={job_resources['scratch_local']}

        #PBS -o "{task_folder}/{job_name}.out"
        #PBS -e "{task_folder}/{job_name}.err"

        HOMEDIR="{home_dir}"
        JOBDIR="{task_folder}"
        CONFIG="{config_path}"
        SOURCE_FILE="{python_script_path}"
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

        job_script_path = os.path.join(self.jobs_dir, job_name, f"{job_name}.sh")
        with open(job_script_path, 'w', newline='\n') as file:
            file.write(job_script)

        return job_script_path

    def _prepare_remote_environment(self, job_name, local_bash_script_path, local_config_path):
        """
        Prepare the remote environment for job submission:
        - Create the task specific directory.
        - Copy the job script and config.
        """
        # Create the task specific remote paths
        task_folder = "/".join((self.remote_work_dir, 'jobs', job_name))
        remote_python_script_path = "/".join((task_folder, os.path.basename(local_bash_script_path)))
        remote_config_path = "/".join((task_folder, os.path.basename(local_config_path)))

        # Create folder and upload files
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        sftp = None
        try:
            ssh.connect(self.remote_host, username=self.username, password=os.getenv('SSH_PASSWORD'))
            sftp = ssh.open_sftp()

            for folder in ["/".join((self.remote_work_dir, 'jobs')), task_folder]:
                try:
                    sftp.stat(folder)
                    print(f"Remote directory {folder} already exists")
                except IOError:
                    print(f"Remote directory {folder} does not exist, creating it")
                    sftp.mkdir(folder)

            for local_filepath, remote_filepath in [(local_bash_script_path, remote_python_script_path),
                                                    (local_config_path, remote_config_path)]:
                sftp.put(local_filepath, remote_filepath)
                print(f"Successfully copied {local_filepath} to {remote_filepath}")

                # Verify the file exists in the remote location
                try:
                    sftp.stat(remote_filepath)
                    print(f"File {remote_filepath} exists on remote server")
                except IOError:
                    print(f"File {remote_filepath} does not exist on remote server after copy")

        except paramiko.SSHException as e:
            print(f"SSH error: {str(e)}")
            raise
        except Exception as e:
            print(f"Failed to copy job script: {str(e)}")
            raise
        finally:
            if sftp:
                sftp.close()
            if ssh:
                ssh.close()

        return remote_python_script_path, remote_config_path

    def _download_log_files(self, remote_files, local_files):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        sftp = None
        try:
            ssh.connect(self.remote_host, username=self.username, password=os.getenv('SSH_PASSWORD'))
            sftp = ssh.open_sftp()

            for remote_filepath, local_filepath in zip(remote_files, local_files):
                sftp.get(remote_filepath, local_filepath)
                print(f"Successfully downloaded {remote_filepath} to {local_filepath}")

                # Verify the file exists in the local location
                if os.path.exists(local_filepath):
                    print(f"File {local_filepath} exists locally")
                else:
                    print(f"File {local_filepath} does not exist locally after download")

        except paramiko.SSHException as e:
            print(f"SSH error: {str(e)}")
            raise
        except Exception as e:
            print(f"Failed to download files: {str(e)}")
            raise
        finally:
            if sftp:
                sftp.close()
            if ssh:
                ssh.close()

    def _submit_job(self, job_script_path):
        """Submit a job to the PBS queue and return the job ID."""

        # Submit the job script
        ssh_command = f'qsub {job_script_path}'
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(self.remote_host, username=self.username, password=os.getenv('SSH_PASSWORD'))
            stdin, stdout, stderr = ssh.exec_command(ssh_command)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status == 0:
                job_id = stdout.read().decode().strip()
                print(f"Job {job_id} submitted successfully.")
                return job_id
            else:
                error_message = stderr.read().decode().strip()
                print(f"Failed to submit job: {error_message}")
                raise Exception(f"Failed to submit job: {error_message}")
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            raise
        finally:
            ssh.close()

    def _log_job(self, job_id, job_name, s3_directory, job_output_path, job_script, config_path, resources):
        """Log job submission details to the SQLite database."""
        # Insert data into the table
        job_data = {
            "job_id": job_id,
            "job_name": job_name,
            "processed_s3_directory": s3_directory,
            "output_directory": job_output_path,
            "batch_script": job_script,
            "python_script": self.python_script_path,
            "config_data": config_path,
            "host_name": "PBS_HOST",
            "requested_resources": str(resources),
            "job_status": "SUBMITTED"
        }
        self.database_manipulator.insert("jobs", job_data, use_transaction=True)
        self.database_manipulator.close_connection()

    def submit_jobs(self,
                    bucket_name: str,
                    directory_filter: str = "",
                    directories: Optional[Union[List, Tuple]] = [],
                    job_config: Optional[Dict] = None,
                    job_config_path: Optional[str] = None,
                    python_script_path: Optional[str] = None,
                    use_gpu: bool = False,
                    resources: Optional[Dict] = None,
                    username: Optional[str] = None,
                    remote_host: Optional[str] = None,
                    ssh_password: str = None,
                    ignore_ssh_auth: bool = False):
        """Submit a job for each directory in the S3 bucket."""

        # Assign attributes
        self.bucket_name = bucket_name
        self.directories = directories
        self.python_script_path = python_script_path
        self.use_gpu = use_gpu
        self.resources = resources if resources else {}
        self.config_path = job_config_path
        self.config = self.load_config() if not job_config else job_config
        self.username = username
        self.remote_host = remote_host

        if Manipulator.is_valid_regex(directory_filter):
            self.directory_filter = directory_filter
        else:
            raise ValueError(f"Invalid regex pattern: {directory_filter}")

        # Get directories based on user input
        s3_directories = self._get_directories()

        # Auth
        # Set the SSH_PASSWORD environment variable
        if ssh_password:
            os.environ['SSH_PASSWORD'] = ssh_password
        try:
            SSHHandler.authenticate(self.username, ssh_password, self.remote_host, os.path.join(self.local_work_dir, '.ssh', 'id_rsa'))
        except Exception as e:
            if not ignore_ssh_auth:
                raise RuntimeError("SHH authentication failed.") from e

        for s3_directory in s3_directories:
            job_name = f"{self.bucket_name}_{s3_directory.rstrip('/').split('/')[-1]}"
            job_local_output_path = Manipulator.create_folders([job_name], self.jobs_dir)[0]

            # Determine the requested resources for the job
            job_resources = self._determine_resources()

            # Generate the job config file
            job_config_path = self._generate_config_file(job_name, self._get_config_updates(s3_directory))

            # Generate the job bash script
            job_bash_script_path = self._generate_job_script(job_name=job_name, job_resources=job_resources, python_script_path=self.python_script_path, config_path=job_config_path)

            # If run remotely prepare the remote environment
            if self._is_remote:
                remote_job_bash_script_path, remote_job_config_path = self._prepare_remote_environment(job_name=job_name, local_bash_script_path=job_bash_script_path, local_config_path=job_config_path)
            else:
                remote_job_bash_script_path, remote_job_config_path = None, None

            # Submit the job using either local or remote path
            job_id = self._submit_job(job_bash_script_path if not self._is_remote else remote_job_bash_script_path)

            # We log the local paths so that they can be retrieved for analysis locally later
            self._log_job(job_id, job_name, s3_directory, job_local_output_path, job_bash_script_path, job_config_path, job_resources)

    def monitor_jobs(self,
                     user_email: str,
                     s3_cfg_file: str = S3_CONFIG,
                     llm_handler=None,
                     username: Optional[str] = None,
                     remote_host: Optional[str] = None,
                     ssh_password: str = None,
                     ignore_ssh_auth: bool = False):
        """Monitor the status of submitted jobs, updating the database accordingly."""

        self.username = username
        self.remote_host = remote_host

        # Auth
        # Set the SSH_PASSWORD environment variable
        if ssh_password:
            os.environ['SSH_PASSWORD'] = ssh_password
        try:
            SSHHandler.authenticate(self.username, ssh_password, self.remote_host, os.path.join(self.local_work_dir, '.ssh', 'id_rsa'))
        except Exception as e:
            if not ignore_ssh_auth:
                raise RuntimeError("SHH authentication failed.") from e

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
                info = get_job_info(job[0], self.username, self.remote_host)
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
                            if self._is_remote:
                                out_local_path = os.path.join(self.jobs_dir, job[1], f'{job[1]}.out')
                                err_local_path = os.path.join(self.jobs_dir, job[1], f'{job[1]}.err')
                                self._download_log_files((info['output_path'], info['error_path']),
                                                         (out_local_path, err_local_path))
                                info['output_path'] = out_local_path
                                info['error_path'] = err_local_path
                        except Exception as e:
                            logging.error(f"Error downloading log files: {e}")

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

