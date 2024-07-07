import logging
import subprocess
import sys
import importlib.util
import importlib.metadata
import os
import site


def check_and_install_package(package_name):
    try:
        importlib.util.find_spec(package_name)
        logging.info(f"{package_name} package is installed.")
    except ImportError:
        logging.warning(f"{package_name} package not found. Installing...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
            logging.info(f"{package_name} package installed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install {package_name} package: {e}")
            sys.exit(1)


def get_detectflow_requirements():
    try:
        import detectflow
    except ImportError:
        logging.error("detectflow package not found.")
        sys.exit(1)

    detectflow_location = os.path.dirname(os.path.dirname(detectflow.__file__))
    requirements_path = os.path.join(detectflow_location, 'requirements.txt')

    if not os.path.exists(requirements_path):
        logging.error(f"requirements.txt not found in detectflow module at {requirements_path}.")
        sys.exit(1)

    with open(requirements_path, 'r') as file:
        required_packages = file.read().splitlines()

    return required_packages


def validate_environment(required_packages):
    logging.info("Validating environment...")

    for package in required_packages:
        package_name = package.split('==')[0]  # Extract package name from version specifier
        if package_name and not package_name.startswith("#"):
            check_and_install_package(package_name)


def check_cuda_availability():
    # Check for CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            logging.info("CUDA is available.")
        else:
            logging.info("CUDA is not available.")
    except ImportError:
        logging.warning("PyTorch is not installed, CUDA availability cannot be determined.")


def setup_environment():
    package_name = 'git+https://github.com/ChlupacTheBosmer/DetectFlow.git@main#egg=DetectFlow'

    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        logging.info(f"{package_name} package installed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install {package_name} package: {e}")
        sys.exit(1)

    # Get the site-packages directory
    site_packages_dir = site.getusersitepackages()
    os.environ["PATH"] += os.pathsep + site_packages_dir
    globals()["detectflow"] = __import__("detectflow")

    required_packages = get_detectflow_requirements()
    validate_environment(required_packages)
    check_cuda_availability()


if __name__ == "__main__":
    setup_environment()
