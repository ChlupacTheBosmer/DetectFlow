import logging
import subprocess
import importlib
import sys
import os
import venv


def create_virtualenv(env_name='venv'):
    """Create a virtual environment if it doesn't exist."""
    if not os.path.exists(env_name):
        venv.create(env_name, with_pip=True)
        logging.info(f"Virtual environment '{env_name}' created successfully.")
    else:
        logging.info(f"Virtual environment '{env_name}' already exists.")


def activate_virtualenv(env_name='venv'):
    """Activate the virtual environment."""
    activate_script = os.path.join(env_name, 'Scripts', 'activate_this.py') if os.name == 'nt' else os.path.join(
        env_name, 'bin', 'activate_this.py')
    with open(activate_script) as file:
        exec(file.read(), dict(__file__=activate_script))


def install_package(package_name, no_deps=False):
    """Install a package using pip."""
    command = [sys.executable, '-m', 'pip', 'install', package_name]
    if no_deps:
        command.append('--no-deps')
    try:
        subprocess.check_call(command)
        logging.info(f"{package_name} package installed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install {package_name} package: {e}")
        sys.exit(1)


def uninstall_package(package_name):
    """Uninstall a package using pip."""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', package_name])
        logging.info(f"{package_name} package uninstalled successfully.")
    except subprocess.CalledProcessError as e:
        logging.warning(f"{package_name} package was not installed: {e}")


def get_detectflow_requirements():
    """Clone the DetectFlow repository and read requirements.txt."""
    temp_dir = "temp_detectflow"
    if os.path.exists(temp_dir):
        subprocess.check_call(['rm', '-rf', temp_dir])
    subprocess.check_call(
        ['git', 'clone', '--depth', '1', 'https://github.com/ChlupacTheBosmer/DetectFlow.git', temp_dir])
    requirements_path = os.path.join(temp_dir, 'requirements.txt')
    if not os.path.exists(requirements_path):
        logging.error(f"requirements.txt not found in detectflow repository at {requirements_path}.")
        return []

    with open(requirements_path, 'r') as file:
        required_packages = file.read().splitlines()

    subprocess.check_call(['rm', '-rf', temp_dir])
    return required_packages


def validate_environment(required_packages):
    """Validate and install missing packages."""
    logging.info("Validating environment...")

    for package in required_packages:
        package_name = package.split('==')[0]  # Extract package name from version specifier
        if package_name and not package_name.startswith("#"):
            try:
                importlib.util.find_spec(package_name)
                logging.info(f"{package_name} package is installed.")
            except ImportError:
                logging.warning(f"{package_name} package not found. Installing...")
                install_package(package)

def check_cuda_availability():
    """Check for CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            logging.info("CUDA is available.")
        else:
            logging.info("CUDA is not available.")
    except ImportError:
        logging.warning("PyTorch is not installed, CUDA availability cannot be determined.")


def setup_environment():
    """Setup the environment."""
    create_virtualenv()
    activate_virtualenv()

    # Update pip to the latest version
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        logging.info("Pip upgraded successfully.")
    except subprocess.CalledProcessError as e:
        logging.warning(f"Failed to upgrade pip: {e}")

    # Uninstall DetectFlow if it's already installed
    uninstall_package('detectflow')

    # Install DetectFlow without dependencies
    install_package('git+https://github.com/ChlupacTheBosmer/DetectFlow.git@main#egg=DetectFlow', no_deps=True)

    # Validate and install missing dependencies
    required_packages = get_detectflow_requirements()
    validate_environment(required_packages)

    check_cuda_availability()


if __name__ == "__main__":
    setup_environment()

