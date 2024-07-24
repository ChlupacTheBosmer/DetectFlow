import json
from tqdm import tqdm
import traceback
from detectflow.config import TESTS_DIR
from detectflow.manipulators.database_manipulator import DatabaseManipulator
from detectflow.manipulators.dataloader import Dataloader
from detectflow.utils.benchmark import DetectionBenchmarker
import os

def get_db_filepaths(directory):
    """Get all .db filepaths from the specified directory, excluding subdirectories."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.db') and os.path.isfile(os.path.join(directory, f))]

def fill_visit_ids(directory_path: str):
    """
    Fill the visit_ids column with -1.0 for each row where visit_ids is an empty list.

    Args:
        directory_path (str): The directory path where the .db files are located.
    """

    # Get all .db filepaths
    db_filepaths = get_db_filepaths(directory_path)

    # Print the filepaths
    with tqdm(total=len(db_filepaths), desc="Processing databases") as pbar:
        for db_path in db_filepaths:

            # Update the description
            pbar.set_description(f"Preparing data - {os.path.basename(db_path)}")

            # Connect to the SQLite database
            db_man = DatabaseManipulator(db_path)

            # Query to select all rows from the visits table
            query = "SELECT frame_number, video_id, all_visitor_bboxes, relevant_visitor_bboxes, visit_ids FROM visits"
            rows = db_man.fetch_all(query)

            # Iterate through each row
            data_entries = []
            for row in rows:
                frame_number, video_id, all_visitor_bboxes_json, relevant_visitor_bboxes_json, visit_ids_json = row

                # Load JSON data
                all_visitor_bboxes = json.loads(all_visitor_bboxes_json)
                visit_ids = json.loads(visit_ids_json)

                # Check if visit_ids is an empty list
                if visit_ids == []:
                    # Create a new visit_ids list with -1 for each list in all_visitor_bboxes
                    new_visit_ids = [-1.0 for _ in all_visitor_bboxes]

                    # Convert new_visit_ids to JSON format
                    new_visit_ids_json = json.dumps(new_visit_ids)

                    data_entries.append((new_visit_ids_json, frame_number, video_id))

            # Update the description
            pbar.set_description(f"Inserting data - {os.path.basename(db_path)}")

            if len(data_entries) > 0:
                query = "UPDATE visits SET visit_ids = ? WHERE frame_number = ? AND video_id = ?"
                db_man.safe_execute(query, data_entries)

            # Commit the changes and close the connection
            db_man.close_connection()

            # Update the progress bar
            pbar.update(1)


def fetch_excels(directory_path: str):
    """
    Fetch the excel filepaths from the specified directory.

    Args:
        directory_path (str): The directory path where the excel files are located.
    """
    db_filepaths = get_db_filepaths(directory_path)

    dat = Dataloader()

    for db_path in db_filepaths:
        db_name = os.path.splitext(os.path.basename(db_path))[0]
        file = dat.locate_file_s3(f"{db_name}", "excels")
        if file is None:
            print(f"Excel file for {db_name} not found.")
        else:
            print(f"Excel file for {db_name} found at {file}")
            dat.download_file_s3(*Dataloader.parse_s3_path(file), os.path.join(directory_path, f"{db_name}.xlsx"))
            print(f"Excel file for {db_name} downloaded successfully.")


def benchmark_db(db_path: str, excel_path: str, checkpoint_file: str, process_all: bool = False):
    """
    Benchmark the detection results for the specified database.

    Args:
        db_path (str): The path to the .db file.
        excel_path (str): The path to the .xlsx file.
        checkpoint_file (str): The path to the checkpoint file.
    """
    # Create output directory
    output_dir = os.path.join(os.path.dirname(db_path), "plots", os.path.splitext(os.path.basename(db_path))[0])
    os.makedirs(output_dir, exist_ok=True)

    # Get frame skip
    if not process_all:
        user_input = input(
            f"Enter frame skip as detections ran for {os.path.splitext(os.path.basename(db_path))[0]}? (int): ").strip().lower()
    else:
        user_input = '15'

    try:
        frame_skip = int(user_input)
    except ValueError:
        frame_skip = 15

    # Run benchmarker
    try:
        benchmarker = DetectionBenchmarker(db_path=db_path, excel_path=excel_path, frame_skip=frame_skip, checkpoint_file=checkpoint_file)
    except Exception as e:
        raise RuntimeError(f"An error occurred while initializing the DetectionBenchmarker: {e}") from e
    try:
        benchmarker.get_ground_truth()
    except Exception as e:
        raise RuntimeError(f"An error occurred while getting the ground truth: {e}") from e
    try:
        all_results, rel_results = benchmarker.benchmark(min_detections=1, frame_gap=frame_skip, output_dir=output_dir)
    except Exception as e:
        raise RuntimeError(f"An error occurred while benchmarking the database: {e}") from e

    # Print results to console
    print("All results:")
    if all_results is not None:
        for key, value in all_results.items():
            if isinstance(value, dict):
                print(f'{key}:')
                for k, v in value.items():
                    print(f'  {k}: {v}')
            else:
                print(f'{key}: {value}')

    try:
        print(f"All: Detected {len(all_results['detected_visits'])} out of {len(all_results['visits'])} visits")
    except Exception:
        print("Detected visits could not be calculated.")

    print("Relevant results:")
    if rel_results is not None:
        for key, value in rel_results.items():
            if isinstance(value, dict):
                print(f'{key}:')
                for k, v in value.items():
                    print(f'  {k}: {v}')
            else:
                print(f'{key}: {value}')

    try:
        print(f"Relevant: Detected {len(rel_results['detected_visits'])} out of {len(rel_results['visits'])} visits")
    except Exception:
        print("Detected visits could not be calculated.")

    # Save results to a file
    results_file_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_file_path, 'w') as results_file:
        if all_results is not None:
            results_file.write("All results:\n")
            for key, value in all_results.items():
                if isinstance(value, dict):
                    results_file.write(f'{key}:\n')
                    for k, v in value.items():
                        results_file.write(f'  {k}: {v}\n')
                else:
                    results_file.write(f'{key}: {value}\n')
        if rel_results is not None:
            results_file.write("Relevant results:\n")
            for key, value in rel_results.items():
                if isinstance(value, dict):
                    results_file.write(f'{key}:\n')
                    for k, v in value.items():
                        results_file.write(f'  {k}: {v}\n')
                else:
                    results_file.write(f'{key}: {value}\n')

        results_file.write(f"All: Detected {len(all_results['detected_visits'])} out of {len(all_results['visits'])} visits\n")
        results_file.write(f"Relevant: Detected {len(rel_results['detected_visits'])} out of {len(rel_results['visits'])} visits")

    print(f"Benchmark results saved to {results_file_path}")


def benchmark_dbs(directory_path: str, process_all: bool = False):
    """
    Benchmark the detection results for the databases in the specified directory.

    Args:
        directory_path (str): The directory path where the .db files are located.
    """
    db_filepaths = get_db_filepaths(directory_path)

    for db_path in db_filepaths:
        db_name = os.path.splitext(os.path.basename(db_path))[0]

        if not process_all:
            user_input = input(f"Do you want to benchmark, database {db_name}? (y/n): ").strip().lower()
        else:
            user_input = 'y'

        if user_input == 'y':
            excel_path = os.path.join(directory_path, f"{db_name}.xlsx")
            checkpoint_file = os.path.join(directory_path, f"{db_name}_checkpoint.json")
            try:
                benchmark_db(db_path, excel_path, checkpoint_file, process_all)
            except Exception as e:
                print(f"An error occurred while benchmarking database {db_name}: {e}")
                traceback.print_exc()
        elif user_input == 'n':
            print(f"Skipped database {db_name}")
        else:
            print("Invalid input, skipping this database.")


if __name__ == "__main__":
    dir_path = os.path.join(TESTS_DIR, 'scratch')
    #fill_visit_ids(dir_path)
    #fetch_excels(dir_path)
    benchmark_dbs(dir_path, True)












