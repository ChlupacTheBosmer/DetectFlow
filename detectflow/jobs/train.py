import torch.cuda
import os
import argparse
from ultralytics import YOLO
from ultralytics import settings
import torch
import clearml
from clearml import Task
import json

def get_gpu_memory_cuda():
    """
    Returns the total memory of the current GPU in GB.

    Requires a machine with a CUDA-capable GPU. Initializes the GPU and queries its total memory.

    Returns:
        float: Total GPU memory in GB.

    Raises:
        RuntimeError: If no CUDA-capable GPU is found.
    """
    if torch.cuda.is_available():
        torch.cuda.init()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        # Convert bytes to GB
        return gpu_memory / (1024**3)
    else:
        raise RuntimeError("No GPU found. Make sure to run this on a machine with a GPU.")

def calculate_batch_size(memory_gb, divisor=0.4):
    """
    Calculates the batch size for training based on the GPU memory.

    Args:
    - memory_gb (float): Total GPU memory in GB.
    - divisor (float): The divisor for calculating batch size. Default is 0.2.

    Returns:
    - int: Calculated batch size.
    """
    return int(memory_gb / divisor)

def get_node_name(hostname):
    """
        Extracts the node name from the hostname.

        Args:
            hostname (str): The hostname to extract the node name from.

        Returns:
            str: Node name extracted from the hostname.
    """
    # Extracts the node name from the hostname (e.g., "fau2" from "fau2.natur.cuni.cz")
    return hostname.split('.')[0]

def get_gpu_memory(node_name, gpu_memory_dict):
    """
        Retrieves the GPU memory size for the given node name and converts it to GB.

        Args:
            node_name (str): Name of the node whose memory size is to be retrieved.
            gpu_memory_dict (dict): Dictionary mapping node names to their GPU memory in MB.

        Returns:
            float: GPU memory size in GB for the given node name.
    """
    # Retrieves the GPU memory size for the given node name and converts it to GB
    memory_mb = gpu_memory_dict.get(node_name)
    if memory_mb is None:
        return 0
    return memory_mb / 1024

def initialize_cuda_settings():
    """
        Configures CUDA environment variables for optimal performance during training.

        Sets specific environment variables related to CUDA operations.
    """
    # Specify some CUDA setttings
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

def initialize_clearml():
    """
        Initializes ClearML environment variables for task monitoring and management.

        Sets up necessary environment variables for ClearML integration, enabling task monitoring
        and management in a remote server setup.
    """
    # Init ClearML
    os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml'
    os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'
    os.environ['CLEARML_FILES_HOST'] = 'https://files.clear.ml'
    os.environ['CLEARML_API_ACCESS_KEY'] = 'VHXADOOTOSMBAGRU5E4T'
    os.environ['CLEARML_API_SECRET_KEY'] = 'TCOeMkjTwyXGnWiFEgYs9nvWeiHHtGiJ2iwk8VhJo7M1CpiYTe'

def initialize_yolo_settings(datasets_dir: str, runs_dir: str):
    """
        Configures settings for YOLO datasets and run results directories.

        Creates and sets up the specified directories for storing datasets and results of YOLO model runs.

        Args:
            datasets_dir (str): Directory for storing datasets.
            runs_dir (str): Directory for storing run results.
    """
    # Create the directories
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    # Update YOLO native settings file to look for datasets and store results in custom dirs
    settings.update({'datasets_dir': datasets_dir, 'runs_dir': runs_dir})

def assign_device():
    """
        Detects available CUDA devices and assigns one or more for training.

        Based on the number of available CUDA devices, this function assigns either a single GPU,
        multiple GPUs, or falls back to CPU for training.

        Returns:
            str or list: Indicating the selected CUDA device(s) or 'cpu' if no CUDA device is available.
    """
    # Detect CUDA devices
    num_cuda_devices = torch.cuda.device_count()

    # Assign value to the device selection variable
    if num_cuda_devices == 1:
        device = "0"
    elif num_cuda_devices > 1:
        device = [int(i) for i in range(num_cuda_devices)]
    else:
        device = "cpu"

    print(f"CUDA available: {torch.cuda.is_available()}, Using CUDA device(s): {device}")

    return device

def assign_workers():
    """
        Calculates the number of workers to use for data loading based on available CPU cores and CUDA devices.

        Determines the optimal number of workers for data loading by dividing the total number of CPU cores
        by the number of CUDA devices.

        Returns:
            int: Number of workers for data loading.
    """
    # Detect CUDA devices
    num_cuda_devices = torch.cuda.device_count()

    # Assign value to the number of workers based on the number of cores used per GPU
    num_cpu_cores = os.cpu_count()

    num_workers = num_cpu_cores // max(1, num_cuda_devices)

    print(f"Number of CPU cores: {os.cpu_count()}, Using workers: {num_workers}")

    return num_workers

def assign_batch_size(method: int = 0, hostname = None):
    """
        Determines the batch size for training based on the available GPU memory.

        Depending on the method chosen, it either dynamically detects GPU memory or retrieves it from a pre-defined dictionary.

        Args:
            method (int): Method for detecting GPU memory. 0 for dynamic detection, 1 for dictionary-based.
            hostname (str): Hostname of the machine, used in method 1 for memory retrieval.

        Returns:
            int: Calculated batch size for training.
    """
    # Define node dictionary
    gpu_memory_by_node = {
        # adan cluster nodes
        **{f"adan{i}": 15109 for i in range(1, 62)},
        # black cluster nodes
        "black1": 16280,
        # galdor cluster nodes
        **{f"galdor{i}": 45634 for i in range(1, 21)},
        # glados cluster nodes
        "glados1": 12066,
        **{f"glados{i}": 7982 for i in range(2, 8)},
        **{f"glados{i}": 11178 for i in range(11, 14)},
        # luna cluster nodes
        **{f"luna{i}": 45634 for i in range(201, 207)},
        # fer cluster nodes
        **{f"fer{i}": 16117 for i in range(1, 4)},
        # zefron cluster nodes
        "zefron6": 22731,
        "zefron7": 8119,
        "zefron8": 11441,
        # zia cluster nodes
        **{f"zia{i}": 40536 for i in range(1, 6)},
        # fau cluster nodes
        **{f"fau{i}": 16125 for i in range(1, 4)},
        # cha cluster nodes
        "cha": 11019,
        # gita cluster nodes
        **{f"gita{i}": 11019 for i in range(1, 8)},
        # konos cluster nodes
        **{f"konos{i}": 11178 for i in range(1, 9)},
        # grimbold cluster node
        "grimbold": 12198,
    }

    # Detect CUDA devices
    num_cuda_devices = torch.cuda.device_count()

    try:
        if method == 0:

            # Get GPU memory
            gpu_memory_gb = get_gpu_memory_cuda()

        elif method == 1 and hostname is not None:

            hostname = args.hostname
            node_name = get_node_name(hostname)
            gpu_memory_gb = get_gpu_memory(node_name, gpu_memory_by_node)

        else:
            print("Wrong detection method argument value")
            return 0

        # Calculate batch size
        batch_size_per_gpu = calculate_batch_size(gpu_memory_gb)

        # Calculate batch size
        batch_size = batch_size_per_gpu * max(1, num_cuda_devices)

        print(f"Batch size per GPU: {batch_size_per_gpu}, Using batch size: {batch_size}")
    except Exception as e:
        print(f'Unable to determine batch_size: {e}')
        batch_size = 0

    return batch_size

def get_folder_names(directory):
    """
        Retrieves a list of folder names within a specified directory.

        This function lists all directories within the specified directory, excluding 'Model training frames'.

        Args:
            directory (str): Path to the directory from which folder names will be listed.

        Returns:
            List[str]: A list of folder names found in the specified directory.
    """
    # Get a list of all folder names in the given directory to get dataset folders. Manually excluding the source data.
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and name != 'Model training frames']

def update_dataset_paths(folder: str, datasets_dir: str):
    """
        Updates file paths in dataset YAML and TXT files to reflect directory changes.

        Modifies the dataset paths in the YAML and TXT files of a specified dataset folder,
        to reflect any changes in directory structure or locations.

        Args:
            folder (str): The dataset folder whose paths need updating.
            datasets_dir (str): Path to the datasets directory.
    """

    def update_txt_file(txt_file_path: str):

        # Read the contents of the file
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()

        # Replace "storage" with "auto" in each path
        modified_lines = [line.replace('storage', args.path_update) for line in lines]

        # Split the path and the extension
        file_path, file_extension = os.path.splitext(txt_file_path)

        # Append '_updated' to the file name
        new_txt_file_path = file_path + '_updated' + file_extension

        # Save the modified paths back to the file
        with open(new_txt_file_path, 'w') as file:
            file.writelines(modified_lines)

        print(f"File paths have been updated in {txt_file_path}")

    def update_yaml_file(yaml_file_path: str):
        import yaml

        # Load the YAML file
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Modify the paths in the YAML data
        data['train'] = data['train'].replace('storage', args.path_update)
        data['val'] = data['val'].replace('storage', args.path_update)

        # Split the path and the extension of the YAML file
        file_path, file_extension = os.path.splitext(yaml_file_path)

        # Append '_updated' to the file name
        new_yaml_file_path = file_path + '_updated' + file_extension

        # Save the modified data back to a new YAML file
        with open(new_yaml_file_path, 'w') as file:
            yaml.dump(data, file)

        print(f"YAML file paths have been updated in {new_yaml_file_path}")

    # Define the path to the original YAML file
    yaml_file_path = os.path.join(datasets_dir, folder, f"{folder}.yaml")

    # Change the paths in yaml
    update_yaml_file(yaml_file_path)

    for ds in ["train", "val"]:
        # Define the path to the text file
        txt_file_path = os.path.join(datasets_dir, folder, "images", ds, f"{ds}.txt")

        # Change the paths in txt
        update_txt_file(txt_file_path)


def main(args):
    """
        Main function to run YOLOv8 training with specified configurations.

        This function sets up the training environment, initializes CUDA and ClearML settings, and
        trains a YOLOv8 model based on the arguments provided. It supports dynamic batch size calculation,
        multiple GPUs, and dataset path adjustments.

        Args:
            args: Parsed command-line arguments.
    """

    # Review args
    print(f"Received arguments: {args}")

    # Set some CUDA environment variables
    initialize_cuda_settings()

    # Initialize clearML auth variables
    initialize_clearml()

    # Initiate task
    name = args.dataset if args.name is None else args.name
    task = Task.init(project_name=args.project_name, task_name=name)

    # Get working directory if not specified
    if args.workdir is None:

        # Get current directory (from where the script is being run) - this should be the homedir
        user = os.getenv('USER')
        current_directory = f"/storage/brno2/home/{user}/"

    else:

        # Assigns value specified by the user
        current_directory = args.workdir
    print(f"Current directory: {current_directory}")

    # Get datasets directory
    if args.datasets_dir is None:

        datasets_dir = os.path.join(current_directory, 'datasets')

    else:

        datasets_dir = args.datasets_dir

    print(f"Datasets directory set to: {datasets_dir}")

    # Get runs directory
    runs_dir = os.path.join(current_directory, 'runs')
    print(f"Runs directory set to: {runs_dir}")

    # Update multiple settings
    initialize_yolo_settings(datasets_dir, runs_dir)

    # Get list of datasets
    if args.dataset is None:
        return

    # Paths will be updated to reflect the binding of directories or any necessary modifications
    if args.path_update is not None:
        # Update paths
        update_dataset_paths(args.dataset, datasets_dir)

    # Get devices for training
    device = assign_device()

    # Get number fo workers
    if args.workers is None:
        num_workers = assign_workers()
    else:
        num_workers = 1

    # Get batch size
    if args.batch_size == 0:
        batch_size = assign_batch_size(args.autobatch_method, args.hostname)
        if batch_size == 0:
            batch_size = 32
    else:
        batch_size = args.batch_size

    # Clear cuda cache
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Unable to clear CUDA cache: {e}")

    if args.resume == 0:

        single_cls = True if args.single_cls == 1 else False

        # Load a pretrained model
        model = YOLO(args.model)

        # Put together the path to the dataset data file
        dataset_data = os.path.normpath(os.path.join(datasets_dir, args.dataset, f'{args.dataset}.yaml')) if args.path_update is None else os.path.normpath(os.path.join(datasets_dir, args.dataset, f'{args.dataset}_updated.yaml'))
        print(dataset_data)

        # Define Save directory
        save_dir = os.path.join(current_directory, args.project_name, name)
        # os.makedirs(save_dir, exist_ok=True)
        #
        # # Test save_dir to check where the files will be stored before the training continues
        # # Open a file in write mode
        # with open(os.path.join(save_dir, 'task_test.txt'), 'w') as file:
        #     # Write variable names and values to the file
        #     file.write(f'name: {name}\n')
        #     file.write(f'project_name: {args.project_name}\n')
        #     file.write(f'save_dir: {save_dir}\n')
        #     file.write(f'dataset: {args.dataset}\n')
        #
        # print("File 'test_variables.txt' has been created with the variable data.")


        # Train the model
        try:
            results = model.train(data=dataset_data,
                                  epochs=args.epochs,
                                  batch=batch_size,
                                  imgsz=640,
                                  workers=num_workers,
                                  device=device,
                                  verbose=True,
                                  project=args.project_name,
                                  name=name,
                                  lr0=args.lr0,
                                  lrf=args.lrf,
                                  cos_lr=args.cos_lr,
                                  optimizer=args.optimizer,
                                  single_cls=single_cls,
                                  save_dir=save_dir)
            task.close()
        except:
            raise
    else:
        try:
            if os.path.exists(args.weights):

                # load model from partial weights
                model = YOLO(args.weights)

                # Define Save directory
                save_dir = os.path.join(current_directory, args.project_name, name)
                # os.makedirs(save_dir, exist_ok=True)
                #
                # # Test save_dir to check where the files will be stored before the training continues
                # # Open a file in write mode
                # with open(os.path.join(save_dir, 'task_test.txt'), 'w') as file:
                #     # Write variable names and values to the file
                #     file.write(f'name: {name}\n')
                #     file.write(f'project_name: {args.project_name}\n')
                #     file.write(f'save_dir: {save_dir}\n')
                #     file.write(f'dataset: {args.dataset}\n')
                #
                # print("File 'test_variables.txt' has been created with the variable data.")

                results = model.train(resume=True,
                                      batch=batch_size,
                                      workers=num_workers
                                      )

                task.close()
            else:
                # Raise and return
                raise FileNotFoundError(f'{args.weights} not found')
        except:
            raise

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config.get('arguments', {})

if __name__ == "__main__":


    '''
    This script will run a YOLO8 training and use the arguments passed to it. Ready for use in metacentrum batch script.
    '''

    parser = argparse.ArgumentParser(description='This script will train a YOLOv8 model.')

    # Argument to config file containing the rest of the training arguments
    parser.add_argument('--config', type=str, required=True, help='Path to the config.json file')

    # Arguments defining training
    parser.add_argument('--dataset', type=str, default=None, help='Dataset folder name')
    parser.add_argument('--name', type=str, default=None, help='Task name')
    parser.add_argument('--datasets_dir', type=str, default=None, help='Path to the dataset folder')
    parser.add_argument('--workdir', type=str, default=None, help='Path to the working directory (homedir)')
    parser.add_argument('--path_update', type=str, default=None, help='Path to dataset images and/or labels will be updated by replacing "storage" with a preset path (to reflect binding)')
    parser.add_argument('--project_name', type=str, default="YOLOv8 train", help='Project name')
    parser.add_argument('--model', type=str, default="yolov8n.pt", help='Name or path to a pretrained model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model for')
    parser.add_argument('--batch_size', type=int, default=0, help='Manual override of batch_size, -1 for YOLO AutoBatch')
    parser.add_argument('--autobatch_method', type=int, default=1,
                        help='Method to use for gpu memory detection: 0 - dynamic with torch, 1 - passive from dictionary')
    parser.add_argument('--workers', type=int, default=None, help='Manual override of number of workers')
    parser.add_argument('--lr0', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate (lr0 * lrf)')
    parser.add_argument('--cos_lr', type=bool, default=True, help='Use cos scheduler for learning rate')
    parser.add_argument('--optimizer', type=str, default='auto', help='optimizer: SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto')
    parser.add_argument('--resume', type=int, default=0, help='Whether to resume interrupted training')
    parser.add_argument('--weights', type=str, default="", help='Path to partial weights (last.pt / best.pt)')
    parser.add_argument('--hostname', type=str, default=None, help='$HOSTNAME from the pbs job to extract the node name from')
    parser.add_argument('--single_cls', type=int, default=0, help='Whether to train as single classed model (0/1)')

    # parse arguments
    args = parser.parse_args()

    # Load arguments from JSON config file
    config_args = load_config(args.config)

    # Combine with command-line arguments, giving precedence to config file arguments
    for key, value in config_args.items():
        setattr(args, key, value)

    # Run the main logic with the arguments
    main(args)