import paramiko
import os


def get_job_info(job_id, username, remote_host):
    """Check the status of a job in the PBS system using qstat.
    :param username:
    :param remote_host:
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"Connecting to remote host: {remote_host}")
        ssh.connect(remote_host, username=username, password=os.getenv('SSH_PASSWORD'), timeout=20)

        # Fetching detailed job information
        stdin, stdout, stderr = ssh.exec_command(f'qstat -f {job_id}')
        exit_status = stdout.channel.recv_exit_status()

        if exit_status != 0:
            # If job is not in the current list, try fetching finished job details
            stdin, stdout, stderr = ssh.exec_command(f'qstat -xf {job_id}')
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                return {"status": "UNKNOWN", "message": "Job not found"}

        job_info = stdout.read().decode()
        return parse_qstat_output(job_id, job_info)

    except paramiko.SSHException as e:
        print(f"SSH error: {str(e)}")
        return {"status": "ERROR", "message": str(e)}

    except Exception as e:
        print(f"Error checking job status: {str(e)}")
        return {"status": "ERROR", "message": str(e)}

    finally:
        ssh.close()

    # try:
    #     # Fetching detailed job information
    #     ssh_command = ['ssh', f'{self.username}@{self.remote_host}', 'qstat', '-f', job_id]
    #     result = subprocess.run(ssh_command, capture_output=True, text=True)
    #     if result.returncode != 0:
    #         # If job is not in the current list, try fetching finished job details
    #         ssh_command = ['ssh', f'{self.username}@{self.remote_host}', 'qstat', '-xf', job_id]
    #         result = subprocess.run(ssh_command, capture_output=True, text=True)
    #         if result.returncode != 0:
    #             return {"status": "UNKNOWN", "message": "Job not found"}
    #
    #     # Parse job details
    #     return self.parse_qstat_output(job_id, result.stdout)
    # except Exception as e:
    #     print(f"Error checking job status: {e}")
    #     return {"status": "ERROR", "message": str(e)}


def parse_qstat_output(job_id, output):
    """
    Parse the detailed output from qstat to extract job status and other information.

    Parameters:
        job_id (str): The job ID for which the output was generated.
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

