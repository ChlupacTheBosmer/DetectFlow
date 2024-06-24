from detectflow import Video, Predictor, Inspector
from detectflow.config import ROOT
import random
import traceback
import logging
import os


def gather_flower_data(**kwargs):
    video_path = kwargs.get('video_path', None)
    s3_path = kwargs.get('s3_path', None)
    output_path = kwargs.get('output_path', None)
    inspect = kwargs.get('inspect', False)
    daytime = kwargs.get('daytime', True)

    if not video_path:
        return None

    frame_number = None
    try:
        # Init Video instance
        vid = Video(video_path, reader_method="decord")

        if vid.color_variance > 50 and not daytime:
            logging.warning(f"UNEXPECTED: Day-time video.")
            return None
        elif vid.color_variance <= 50 and daytime:
            logging.warning("UNEXPECTED: Night-time video.")
            return None

        # Define frame numbers to read
        frame_number = random.randint(0, vid.total_frames - 1)

        # Extract frame from a video
        frame = vid.read_video_frame(frame_number, stream=False)[0]['frame']

        # Init Predictor instance
        p = Predictor()

        # Run detection on the frame and store result
        results = []
        for result in p.detect(frame,
                               detection_conf_threshold=0.1,
                               model_path=os.path.join(ROOT, 'models', 'flowers.pt')):

            if result is not None:
                result.source_path = video_path
                result.s3_path = s3_path
                # result.source_name = os.path.basename(video_path)
                result.frame_number = frame_number
                result.save_dir = output_path

                result.save(sort=True, save_txt=True)

                if inspect:
                    Inspector.display_frames_with_boxes(result.orig_img, result.boxes)

            results.append(result)
    except Exception as e:
        logging.error(
            f"Error extracting a frame number: {frame_number} from a video {os.path.basename(video_path)}. Returning None. {e}")
        traceback.print_exc()
        results = None
    return results


if __name__ == "__main__":
    from detectflow import S3Manipulator, Inspector
    from detectflow import VideoDownloader, DetectionResults
    import os
    from functools import partial
    # from IPython.display import Javascript
    import logging
    # user = os.getenv('USER')

    # Set your parameters
    output_path = r"D:\Dílna\Kutění\Python\DetectFlow\tests\temp"
    regex = r"(?<!\.)[^.]*_((0[89]|1[0-8])_[0-6][0-9])\.mp4$" # not starting with dot, time between 08:00 and 18:00
    daytime = True
    sample_size = 3
    clear_output_every = 30

    # Execution logic
    os.makedirs(output_path, exist_ok=True)
    callback = partial(gather_flower_data, output_path=output_path, inspect=True, daytime=daytime)
    s3_manipulator = S3Manipulator()
    buckets = s3_manipulator.list_buckets_s3(regex=r"\b[a-zA-Z]{2,3}\d-[a-zA-Z]\d\b")
    downloader = VideoDownloader(manipulator=s3_manipulator,
                                 checkpoint_file=os.path.join(output_path, "checkpoint.json"),
                                 whitelist_buckets=buckets,
                                 delete_after_process=True,
                                 processing_callback=callback)
    counter = 0
    saved_results = []
    for _, results in downloader.download_videos_random(regex=regex, sample_size=sample_size, parallelism=True):
        if results and isinstance(results, list) and len(results) == 1 and isinstance(results[0], DetectionResults):
            saved_results.append(results[0])
            logging.info(f"Result for video {results[0].source_name} processed.")
        else:
            logging.error(f"Result for video FAILED.")
            logging.info(f"Returned type: {type(results)}")
        counter += 1
        if counter >= clear_output_every:
            # # Clear output to prevent crashing
            # display(Javascript('IPython.notebook.clear_all_output()'))
            counter = 0
    logging.info("All processing finished successfully.")