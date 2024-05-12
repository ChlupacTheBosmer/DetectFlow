import subprocess
import os
from typing import Optional

class SSHHandler:
    '''
    s = SSHHandler()
    p = s.generate_ssh_key('/storage/brno2/home/USER/deploy/.ssh/id_rsa')
    print(p)
    s.copy_ssh_key_to_remote(p, 'USER', 'HOST', 'pass')
    '''

    def __init__(self):

        self.ssh_key_path = None

    def generate_ssh_key(self, ssh_key_path: Optional[str] = None, regenerate: bool = True):

        self.ssh_key_path = ssh_key_path if ssh_key_path is not None else '/.ssh/id_rsa'

        os.makedirs(os.path.dirname(self.ssh_key_path), exist_ok=True)

        try:
            if not os.path.exists(self.ssh_key_path) or regenerate:
                result = subprocess.run(
                    ['ssh-keygen', '-t', 'rsa', '-b', '4096', '-f', ssh_key_path, '-N', ''],
                    input='y',  # Sending 'y' followed by newline to stdin
                    text=True,
                    capture_output=True,
                    check=True
                )
                print("SSH key generated successfully.")
            else:
                print("SSH key already exists.")
            return self.ssh_key_path
        except Exception as e:
            print(f"Error generating SSH key: {e}")
            return None

    def copy_ssh_key_to_remote(self, ssh_key_path: Optional[str] = None, username: Optional[str] = None,
                               remote_host: Optional[str] = None, password: str = None):
        '''
        You can set your password as environment variable: export SSH_PASSWORD="your_secret_password"

        '''

        ssh_key_path = ssh_key_path if ssh_key_path is not None else self.ssh_key_path
        ssh_key_pub_path = f"{ssh_key_path}.pub"
        host_name = f"{username}@{remote_host}"
        password = os.getenv('SSH_PASSWORD') if not password else password

        try:
            if not password:
                raise ValueError(
                    "SSH password environment variable 'SSH_PASSWORD' is not set and no password passed as argument.")
            if ssh_key_path is None:
                raise ValueError("No SSH key path is set.")
            result = subprocess.run(['sshpass', '-p', password, 'ssh-copy-id', '-i', ssh_key_pub_path, host_name],
                                    input='y',  # Sending 'y' followed by newline to stdin
                                    text=True,
                                    capture_output=True,
                                    check=True)
            print("SSH key copied to remote host successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print("Failed to copy SSH key:", e)
            return False
        except ValueError as ve:
            print("Invalid argument value:", ve)
            return False