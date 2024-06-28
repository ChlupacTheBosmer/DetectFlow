from typing import List, Dict, Tuple, Union
from datetime import timedelta
import re

def validate_flags(input_flags: Union[str, int, List[Union[str, int]], Tuple[Union[str, int], ...]],
                   flag_map: Union[Dict[int, str], List[str], Tuple[str, ...]],
                   fix: bool = False) -> List[str]:

    # Check if flag_map is a list or tuple, convert it to a dictionary if so
    if isinstance(flag_map, (list, tuple)):
        flag_map = {i: flag for i, flag in enumerate(flag_map)}
    allowed_flags = set(flag_map.values())

    if isinstance(input_flags, str):
        if input_flags not in allowed_flags:
            if fix:
                return []
            else:
                raise ValueError("Invalid flag string")
        input_flags = [input_flags]
    elif isinstance(input_flags, int):
        if input_flags not in flag_map:
            if fix:
                return []
            else:
                raise ValueError("Invalid flag integer")
        input_flags = [flag_map[input_flags]]
    elif isinstance(input_flags, (list, tuple)):
        new_flags = []
        for flag in input_flags:
            if isinstance(flag, int):
                if flag not in flag_map:
                    if not fix:
                        raise ValueError("Invalid flag integer")
                    continue  # Skip invalid integers if fixing
                new_flags.append(flag_map[flag])
            elif isinstance(flag, str):
                if flag not in allowed_flags:
                    if not fix:
                        raise ValueError("Invalid flag string")
                    continue  # Skip invalid strings if fixing
                new_flags.append(flag)
            else:
                raise TypeError("Each flag should be either an integer or a string")
        input_flags = new_flags
    else:
        raise TypeError("Input flags should be a string, integer, list, or tuple")

    return input_flags


def format_duration(time_input):
    """
    Converts various time formats to a standardized 'hh:mm:ss' format representing duration.

    Parameters:
        time_input (str or int or float): Input time in various formats ('01:30', '1:15', '1', 1, 1.5, etc.)

    Returns:
        str: Time duration in 'hh:mm:ss' format.
    """
    try:
        if isinstance(time_input, (int, float)):  # Direct number input
            # Assuming the number is an hour count, convert to timedelta
            total_seconds = int(time_input * 3600)
        else:
            # Handle string input assuming it could be hours:minutes, hours:minutes:seconds or just hours
            parts = [int(part) for part in re.split("[:]", time_input)]
            if len(parts) == 1:
                # Only hours are provided
                total_seconds = parts[0] * 3600
            elif len(parts) == 2:
                # Hours and minutes are provided
                total_seconds = parts[0] * 3600 + parts[1] * 60
            elif len(parts) == 3:
                # Hours, minutes, and seconds are provided
                total_seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
            else:
                return "Invalid time format"

        # Create a timedelta object from the total seconds
        td = timedelta(seconds=total_seconds)
        # Formatting to 'hh:mm:ss'
        return str(td)
    except ValueError:
        return "Invalid input"
