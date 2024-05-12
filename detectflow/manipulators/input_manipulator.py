import re


class InputManipulator:
    def __init__(self):
        pass

    @staticmethod
    def get_bucket_name_from_id(string: str):
        # Split the string by underscore
        parts = string.split('_')

        # Take the first two parts and join them with a hyphen
        # Also, convert them to lowercase
        if parts:
            formatted_string = f"{parts[0].lower()}-{parts[1].lower()}"
            return formatted_string
        else:
            return None

    @staticmethod
    def escape_string(string):
        # Escape special regex characters in the IDs in case they contain any
        return re.escape(string)

    @staticmethod
    def zero_pad_id(string):
        # This regex pattern captures a group of characters ending with the second underscore
        # and followed by the integer we want to zero-pad
        match = re.search(r'(^[^_]*_[^_]*_[^_]*)(\d+)', string)

        if match:
            prefix, number = match.groups()

            # Check if the number is already zero-padded or if it's less than 10 and not zero-padded
            if len(number) == 1 or (len(number) > 1 and not number.startswith('0')):
                zero_padded_number = number.zfill(2)
                string = prefix + zero_padded_number + string[len(prefix) + len(number):]

        return string