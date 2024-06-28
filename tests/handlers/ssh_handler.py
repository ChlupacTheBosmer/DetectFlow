import os
import unittest
from detectflow.handlers.ssh_handler import SSHHandler  # Ensure SSHHandler class is in the same directory or properly imported
import paramiko

class TestSSHHandler(unittest.TestCase):

    def setUp(self):
        # Set up variables for the tests
        self.ssh_key_path = r"D:\Dílna\Kutění\Python\DetectFlow\tests\handlers\test\id_rsa"
        self.username = 'USER'  # Replace with your remote username
        self.remote_host = "HOST"  # Replace with your remote host
        self.password = 'PASSWORD'  # Ensure this is set in your environment for security

        # Create an instance of SSHHandler
        self.ssh_handler = SSHHandler()

    def test_generate_ssh_key(self):
        # Test SSH key generation
        key_path = self.ssh_handler.generate_key(self.ssh_key_path, regenerate=True)
        self.assertIsNotNone(key_path, "Failed to generate SSH key.")
        self.assertTrue(os.path.exists(key_path), "SSH private key file does not exist.")
        self.assertTrue(os.path.exists(f"{key_path}.pub"), "SSH public key file does not exist.")

    def test_copy_ssh_key_to_remote(self):
        # Ensure SSH key is generated
        key_path = self.ssh_handler.generate_key(self.ssh_key_path, regenerate=True)
        self.assertIsNotNone(key_path, "Failed to generate SSH key.")

        # Test copying SSH key to remote
        success = self.ssh_handler.upload_key(ssh_key_path=key_path, username=self.username,
                                              remote_host=self.remote_host, password=self.password)
        self.assertTrue(success, "Failed to copy SSH key to remote host.")

        # Verify the key is added to the remote authorized_keys
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.remote_host, username=self.username, password=self.password)
        stdin, stdout, stderr = ssh.exec_command('cat ~/.ssh/authorized_keys')

        public_key_content = ""
        with open(f"{key_path}.pub", 'r') as pub_key_file:
            public_key_content = pub_key_file.read().strip()

        authorized_keys = stdout.read().decode().strip()
        ssh.close()

        self.assertIn(public_key_content, authorized_keys, "Public key not found in remote authorized_keys file.")

    def tearDown(self):
        # Clean up generated key files
        # if os.path.exists(self.ssh_key_path):
        #     os.remove(self.ssh_key_path)
        # if os.path.exists(f"{self.ssh_key_path}.pub"):
        #     os.remove(f"{self.ssh_key_path}.pub")

        # Optionally remove the public key from the remote authorized_keys file
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.remote_host, username=self.username, password=self.password)
        ssh.exec_command(f'sed -i "/{self.ssh_key_path}.pub/d" ~/.ssh/authorized_keys')
        ssh.close()

if __name__ == '__main__':
    unittest.main()
