import subprocess
import os
from typing import Optional
import paramiko


class SSHHandler:
    def __init__(self):

        self.ssh_key_path = None

    def generate_key(self, ssh_key_path: Optional[str] = None, regenerate: bool = True):

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

    def upload_key(self, ssh_key_path: Optional[str] = None, username: Optional[str] = None,
                   remote_host: Optional[str] = None, password: str = None):
        '''
        You can set your password as environment variable: export SSH_PASSWORD="your_secret_password"

        '''

        ssh_key_path = ssh_key_path if ssh_key_path is not None else self.ssh_key_path
        ssh_key_pub_path = f"{ssh_key_path}.pub"
        password = os.getenv('SSH_PASSWORD') if not password else password

        try:
            # Read the public key content
            with open(ssh_key_pub_path, 'r') as pub_key_file:
                public_key = pub_key_file.read()

            # Establish SSH connection
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(remote_host, username=username, password=password)

            # Execute the command to add the public key to the authorized_keys
            command = f'echo "{public_key.strip()}" >> ~/.ssh/authorized_keys'
            stdin, stdout, stderr = ssh.exec_command(command)

            # Check for errors
            error = stderr.read().decode()
            if error:
                print(f"Failed to copy SSH key: {error}")
                return False

            print("SSH key copied to remote host successfully.")
            return True
        except Exception as e:
            print(f"Error copying SSH key: {e}")
            return False

    @staticmethod
    def authenticate(username, password, remote_host, key_path):

        # Prepare SSH keys
        ssh_handler = SSHHandler()
        key_path = ssh_handler.generate_key(key_path)
        if key_path is None:
            raise RuntimeError("SSH key generation failed. Authentication process terminated.")
        if not ssh_handler.upload_key(key_path, username, remote_host, password):
            print("SSH key upload failed. Key might already be uploaded. Authentication might not work.")

