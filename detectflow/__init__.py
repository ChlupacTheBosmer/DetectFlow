# Re-exporting submodules and classes for ease of access
from .handlers import CheckpointHandler, ConfigHandler, CustomChatHandler, EmailHandler, JobHandler, OpenAIChatHandler, SSHHandler
from .image import MotionEnrich, MotionEnrichResult, SmartCrop, CropResult
from .manipulators import BoxAnalyser, BoxManipulator, DatabaseManipulator, Dataloader, FrameManipulator, InputManipulator, Manipulator, S3Manipulator, VideoManipulator
from .predict import Ensembler, Predictor, DetectionBoxes, DetectionResults, Tracker
from .process import DatabaseManager, DatasetMotionEnricher, DatasetSourceProcessor, FrameGenerator, FrameGeneratorTask, Orchestrator, Task, Scheduler
from .utils import get_numeric_hash, Inspector, LogFile, PBSJobReport, PDFCreator, log_function_call, profile_function_call, profile_memory, profile_cpu, Sampler, calculate_optimal_threads, profile_threads, manage_threads
from .validators import InputValidator, ObjectDetectValidator, S3Validator, Validator, VideoValidator
from .video import FrameReader, MotionDetector, VideoDiagnoser, VideoFileInteractive, VideoFilePassive, get_grouped_rois_from_frame, get_unique_rois_from_frame, get_text_with_OCR, extract_time_from_text, install_google_api_key

# Package metadata
__version__ = '0.1.0'
__author__ = 'Petr Chlup'
__email__ = 'USER@natur.cuni.cz'

# Defining the public API of the package
__all__ = [
'CheckpointHandler', 'ConfigHandler', 'CustomChatHandler', 'EmailHandler', 'JobHandler', 'OpenAIChatHandler', 'SSHHandler',
'MotionEnrich', 'MotionEnrichResult', 'SmartCrop', 'CropResult',
'BoxAnalyser', 'BoxManipulator', 'DatabaseManipulator', 'Dataloader', 'FrameManipulator', 'InputManipulator', 'Manipulator', 'S3Manipulator', 'VideoManipulator',
'Ensembler', 'Predictor', 'DetectionBoxes', 'DetectionResults', 'Tracker',
'DatabaseManager', 'DatasetMotionEnricher', 'DatasetSourceProcessor', 'FrameGenerator', 'FrameGeneratorTask', 'Orchestrator', 'Task', 'Scheduler',
'get_numeric_hash', 'Inspector', 'LogFile', 'PBSJobReport', 'PDFCreator', 'log_function_call', 'profile_function_call',
'profile_memory', 'profile_cpu', 'Sampler', 'calculate_optimal_threads', 'profile_threads', 'manage_threads',
'InputValidator', 'ObjectDetectValidator', 'S3Validator', 'Validator', 'VideoValidator',
'FrameReader', 'MotionDetector', 'VideoDiagnoser', 'VideoFileInteractive', 'VideoFilePassive',
'get_grouped_rois_from_frame', 'get_unique_rois_from_frame', 'get_text_with_OCR', 'extract_time_from_text',
'install_google_api_key'
]

# Optional initialization code
print(f"DetectFlow version {__version__} by {__author__} initialized.")