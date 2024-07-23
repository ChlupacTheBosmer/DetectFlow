import argparse
import logging
import os
import traceback
import sys
import json
from typing import Optional, Dict, Any

# Now import detectflow package
from detectflow.utils import install_s3_config
from detectflow.process.scheduler import Scheduler
from detectflow.utils.config import load_config, merge_configs

def main(config_path: str, config_format: str, **kwargs):

    # Load configuration and merge with kwargs
    file_config = load_config(config_path, config_format)
    merged_config = merge_configs(file_config, kwargs)

    # Install s3 config if the parameters were passed
    try:
        if all([key in merged_config for key in ['host_base', 'use_https', 'access_key', 'secret_key', 'host_bucket']]):
            install_s3_config(merged_config.get('host_base'),
                              merged_config.get('use_https'),
                              merged_config.get('access_key'),
                              merged_config.get('secret_key'),
                              merged_config.get('host_bucket'))
    except Exception as e:
        logging.error(f"An error occurred when installing S3 config: {e}")

    # Sort out some kwargs
    merged_config['local_work_dir'] = merged_config.get('local_work_dir', os.getcwd())
    merged_config['remote_work_dir'] = merged_config.get('remote_work_dir', None)
    merged_config['database_path'] = merged_config.get('database_path', os.path.join(merged_config['local_work_dir'], 'scheduler.db'))
    merged_config['ssh_password'] = os.getenv('SSH_PASSWORD', None) if merged_config.get('ssh_password') is None else merged_config.get('ssh_password')
    merged_config['ignore_ssh_auth'] = merged_config.get('ignore_ssh_auth', True)
    merged_config['resources'] = {
        'walltime': merged_config.get('walltime', 1),
        'cpus': merged_config.get('ncpus', 5),
        'use_gpu': merged_config.get('use_gpu', False),
        'mem': merged_config.get('memory', 32),
        'scratch': merged_config.get('scratch_size', 2)
    }

    # Keys for each method
    scheduler_keys = [
        'local_work_dir',
        'remote_work_dir',
        'database_path'
    ]
    submit_keys = [
        'bucket_name',
        'directory_filter',
        'directories',
        'job_config',
        'job_config_path',
        'python_script_path',
        'use_gpu',
        'resources',
        'username',
        'remote_host',
        'ssh_password',
        'ignore_ssh_auth'
    ]
    monitor_keys = [
        'user_email',
        's3_cfg_file',
        'llm_handler',
        'username',
        'remote_host',
        'ssh_password',
        'ignore_ssh_auth'
    ]

    # Sort kwargs into scheduler, submit, and monitor dictionaries
    mode = merged_config.get('mode', 'both')
    scheduler_kwargs = {key: value for key, value in merged_config.items() if key in scheduler_keys}
    submit_kwargs = {key: value for key, value in merged_config.items() if key in submit_keys}
    monitor_kwargs = {key: value for key, value in merged_config.items() if key in monitor_keys}

    # Create Scheduler object
    scheduler = Scheduler(**scheduler_kwargs)

    if mode in ['submit', 'both']:
        try:
            scheduler.submit_jobs(**submit_kwargs)
        except Exception as e:
            logging.error(f"Error submitting jobs: {e}")
            traceback.print_exc()

    if mode in ['monitor', 'both']:
        try:
            scheduler.monitor_jobs(**monitor_kwargs)
        except Exception as e:
            logging.error(f"Error monitoring jobs: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Scheduler with specified configuration.")
    parser.add_argument('--local_work_dir', type=str, help="Path to the local working directory.")
    parser.add_argument('--remote_work_dir', type=str, help="Path to the remote working directory.")
    parser.add_argument('--database_path', type=str, help="Path to the database for logging jobs.")
    parser.add_argument('--config_path', type=str, help="Path to the configuration file for scheduler.")
    parser.add_argument('--config_format', type=str, choices=['json', 'ini'], default='json', help="Configuration file format.")

    # Mode of operation
    parser.add_argument('--mode', type=str, choices=['submit', 'monitor', 'both'], default='submit', help="Operation to perform.")

    # Submit mode arguments
    parser.add_argument('--bucket_name', type=str, help="Name of the bucket to schedule jobs for.")
    parser.add_argument('--directory_filter', type=str, help="Regex for filtering directories.")
    parser.add_argument('--directories', help="List of directories to schedule jobs for.")
    parser.add_argument('--job_config', help="Dictionary with configuration to use for the scheduled jobs.")
    parser.add_argument('--job_config_path', help="Path to a configuration file to use for the scheduled jobs.")
    parser.add_argument('--python_script_path', help="Path to a python script to use for the scheduled jobs.")

    parser.add_argument('--walltime', type=int, default=1, help="Walltime for the scheduled jobs (HH).")
    parser.add_argument('--ncpus', type=int, default=5, help="Number of CPUs to request for the scheduled jobs.")
    parser.add_argument('--use_gpu', type=bool, default=False, help="Whether to request GPU for the scheduled jobs.")
    parser.add_argument('--memory', type=int, default=32, help="Memory to request for the scheduled jobs (GB).")
    parser.add_argument('--scratch_size', type=int, default=2, help="Scratch size to request for the scheduled jobs (GB).")

    # Monitor mode arguments
    parser.add_argument('--user_email', type=str, help="User email for monitoring jobs.")
    parser.add_argument('--s3_cfg_file', type=str, help="Path to the S3 configuration file.")
    parser.add_argument('--llm_handler', help="Handler for the LLM.")

    # Shared arguments
    parser.add_argument('--username', type=str, help="Username for the remote host.")
    parser.add_argument('--remote_host', type=str, help="Remote host for submitting jobs.")
    parser.add_argument('--ssh_password', type=str, help="Password for SSH connection. You can set it as an environment variable 'SSH_PASSWORD'.")
    parser.add_argument('--ignore_ssh_auth', type=bool, help="Whether to ignore SSH authentication.")

    args = parser.parse_args()

    kwargs = {key: value for key, value in vars(args).items() if value is not None}

    print("Arguments (args):", args)
    print("Keyword Arguments (kwargs):", kwargs)

    main(**kwargs)
