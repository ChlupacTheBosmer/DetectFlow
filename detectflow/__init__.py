# Re-exporting submodules and classes for ease of access
from .config import S3_CONFIG
from .handlers.checkpoint_handler import CheckpointHandler
from .handlers.config_handler import ConfigHandler
from .handlers.custom_chat_handler import CustomChatHandler
from .handlers.email_handler import EmailHandler
from .handlers.job_handler import JobHandler
from .handlers.openai_chat_handler import OpenAIChatHandler
from .handlers.ssh_handler import SSHHandler
from .image.motion_enrich import MotionEnrich, MotionEnrichResult
from .image.smart_crop import SmartCrop, CropResult
from .manipulators.box_manipulator import BoxManipulator
from .manipulators.database_manipulator import DatabaseManipulator
from .manipulators.dataloader import Dataloader
from .manipulators.frame_manipulator import FrameManipulator
from .manipulators.input_manipulator import InputManipulator
from .manipulators.manipulator import Manipulator
from .manipulators.s3_manipulator import S3Manipulator
from .manipulators.video_manipulator import VideoManipulator
from .predict.ensembler import Ensembler
from .predict.predictor import Predictor
from .predict.results import DetectionBoxes, DetectionResults
from .predict.tracker import Tracker
from .process.database_manager import DatabaseManager
from .process.dataset_motion_enricher import DatasetMotionEnricher
from .process.dataset_source_processor import DatasetSourceProcessor
from .process.frame_generator import FrameGenerator, FrameGeneratorTask
from .process.orchestrator import Orchestrator, Task
from .process.scheduler import Scheduler
from .process.video_downloader import VideoDownloader
from .utils.extract_data import safe_int, safe_float, safe_str, safe_json, safe_first_element, safe_last_element, safe_datetime, extract_data_from_result, extract_data_from_video
from .utils.file import compare_file_sizes
from .utils.hash import get_numeric_hash
from .utils.inspector import Inspector
from .utils.log_file import LogFile
from .utils.pbs_job_report import PBSJobReport
from .utils.pdf_creator import PDFCreator
from .utils.profile import log_function_call, profile_function_call, profile_memory, profile_cpu
from .utils.sampler import Sampler
from .utils.threads import calculate_optimal_threads, profile_threads, manage_threads
from .validators.object_detect_validator import ObjectDetectValidator
from .validators.s3_validator import S3Validator
from .validators.validator import Validator
from .validators.video_validator import VideoValidator
from .video.frame_reader import SimpleFrameReader, FrameReader
from .video.video_data import Video
from .video.picture_quality import PictureQualityAnalyzer
from .video.motion_detector import MotionDetector
from .video.video_diagnoser import VideoDiagnoser
from .video.video_inter import VideoFileInteractive
from .video.video_passive import VideoFilePassive
from .video.vision_AI import get_grouped_rois_from_frame, get_unique_rois_from_frame, get_text_with_OCR, extract_time_from_text, install_google_api_key


# Package metadata
__version__ = '0.1.0'
__author__ = 'Petr Chlup'
__email__ = 'USER@natur.cuni.cz'


# Defining the public API of the package
__all__ = ['S3_CONFIG',
           'CheckpointHandler',
           'ConfigHandler',
           'CustomChatHandler',
           'EmailHandler',
           'JobHandler',
           'OpenAIChatHandler',
           'SSHHandler',
           'MotionEnrich',
           'MotionEnrichResult',
           'SmartCrop',
           'CropResult',
           'DatabaseManipulator',
           'Dataloader',
           'FrameManipulator',
           'InputManipulator',
           'Manipulator',
           'S3Manipulator',
           'VideoManipulator',
           'Ensembler',
           'Predictor',
           'DetectionBoxes',
           'DetectionResults',
           'Tracker',
           'DatabaseManager',
           'DatasetMotionEnricher',
           'DatasetSourceProcessor',
           'FrameGenerator',
           'FrameGeneratorTask',
           'Orchestrator',
           'Task',
           'Scheduler',
           'VideoDownloader',
           'safe_int',
           'safe_float',
           'safe_str',
           'safe_json',
           'safe_first_element',
           'safe_last_element',
           'safe_datetime',
           'extract_data_from_result',
           'extract_data_from_video',
           'compare_file_sizes',
           'get_numeric_hash',
           'Inspector',
           'LogFile',
           'PBSJobReport',
           'PDFCreator',
           'log_function_call',
           'profile_function_call',
           'profile_memory',
           'profile_cpu',
           'Sampler',
           'calculate_optimal_threads',
           'profile_threads',
           'manage_threads',
           'ObjectDetectValidator',
           'S3Validator',
           'Validator',
           'VideoValidator',
           'SimpleFrameReader',
           'FrameReader',
           'Video',
           'PictureQualityAnalyzer',
           'MotionDetector',
           'VideoDiagnoser',
           'VideoFileInteractive',
           'VideoFilePassive',
           'get_grouped_rois_from_frame',
           'get_unique_rois_from_frame',
           'get_text_with_OCR',
           'extract_time_from_text',
           'install_google_api_key'
           ]


