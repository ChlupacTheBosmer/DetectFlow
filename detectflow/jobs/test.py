import argparse


def main(*args, **kwargs):
    print("Arguments (args):", args)
    print("Keyword Arguments (kwargs):", kwargs)

    # Example operation with args: Summing numeric arguments
    sum_args = sum(arg for arg in args if isinstance(arg, (int, float)))
    print(f"Sum of numeric args: {sum_args}")

    # Example operation with kwargs: Concatenate string values
    concat_kwargs = ''.join(value for value in kwargs.values() if isinstance(value, str))
    print(f"Concatenated string kwargs: {concat_kwargs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple test script.")

    # Add positional arguments
    parser.add_argument('args', nargs='*', help="Positional arguments", type=str)

    # Add optional keyword arguments
    parser.add_argument('--kwargs', nargs='+', help="Keyword arguments in key=value format")

    # Parse arguments
    parsed_args = parser.parse_args()

    # Convert keyword arguments to dictionary
    kwargs = {}
    if parsed_args.kwargs:
        print("Keyword arguments:", parsed_args.kwargs)
        for kwarg in parsed_args.kwargs:
            key, value = kwarg.split('=')
            try:
                # Try to interpret value as an int
                value = int(value)
            except ValueError:
                try:
                    # Try to interpret value as a float
                    value = float(value)
                except ValueError:
                    # Leave value as string
                    pass
            kwargs[key] = value

    # Call the main function with parsed arguments
    main(*parsed_args.args, **kwargs)