import logging
import subprocess
import importlib.util
import sys
import os


def install_package(package_name, version_constraint=None, no_deps=False):
    """Install a package using pip."""
    package_spec = package_name if version_constraint is None else f"{package_name}{version_constraint}"
    command = [sys.executable, '-m', 'pip', 'install', package_spec]
    if no_deps:
        command.append('--no-deps')
    try:
        subprocess.check_call(command)
        logging.info(f"{package_spec} package installed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install {package_spec} package: {e}")
        sys.exit(1)


def uninstall_package(package_name):
    """Uninstall a package using pip."""
    command = [sys.executable, '-m', 'pip', 'uninstall', '-y', package_name]
    try:
        subprocess.check_call(command)
        logging.info(f"{package_name} package uninstalled successfully.")
    except subprocess.CalledProcessError as e:
        logging.warning(f"{package_name} package was not installed: {e}")


def get_installed_packages():
    """Get a list of currently installed packages and their versions."""
    result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=freeze'], stdout=subprocess.PIPE, text=True)
    installed_packages = {}
    for line in result.stdout.split('\n'):
        if '==' in line:
            package, version = line.split('==')
            installed_packages[package] = version
    return installed_packages


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


def validate_environment(required_packages, installed_packages):
    """Validate and install missing packages."""
    logging.info("Validating environment...")

    for package_spec in required_packages:
        if '==' in package_spec:
            package_name, required_version = package_spec.split('==')
        elif '>=' in package_spec:
            package_name, required_version = package_spec.split('>=')
        elif '<=' in package_spec:
            package_name, required_version = package_spec.split('<=')
        else:
            package_name = package_spec
            required_version = None

        if package_name in installed_packages:
            installed_version = installed_packages[package_name]
            if required_version and installed_version != required_version:
                logging.warning(
                    f"{package_name} version {installed_version} is installed, but {required_version} is required. Reinstalling...")
                install_package(package_name, version_constraint=f"=={required_version}")
        else:
            logging.warning(f"{package_name} package not found. Installing...")
            if required_version:
                install_package(package_name, version_constraint=f"=={required_version}")
            else:
                install_package(package_name)


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

    # Get installed packages
    installed_packages = get_installed_packages()

    # Validate and install missing dependencies
    required_packages = get_detectflow_requirements()
    validate_environment(required_packages, installed_packages)

    # Get installed packages
    installed_packages = get_installed_packages()

    # Install specific versions of numpy and urllib3 if not present or different version is required
    if 'urllib3' in installed_packages:
        if not installed_packages['urllib3'].startswith('1.'):
            install_package('urllib3<2.0')
            logging.warning("Urllib3 version is not in the required range. Reinstalling...")
    else:
        install_package('urllib3<2.0')
        logging.warning("Urllib3 version is not in the required range. Reinstalling...")

    if 'opencv-python' in installed_packages:
        if not installed_packages['opencv-python'].startswith('4.7'):
            install_package('opencv-python==4.7.0.72')
            logging.warning("OpenCV version is not in the required range. Reinstalling...")
    else:
        install_package('opencv-python==4.7.0.72')
        logging.warning("OpenCV version is not in the required range. Reinstalling...")

    if 'numpy' in installed_packages:
        if not installed_packages['numpy'].startswith('1.'):
            install_package('numpy>=1.21.6,<1.27')
            logging.warning("Numpy version is not in the required range. Reinstalling...")
    else:
        install_package('numpy>=1.21.6,<1.27')
        logging.warning("Numpy version is not in the required range. Reinstalling...")

    check_cuda_availability()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_environment()