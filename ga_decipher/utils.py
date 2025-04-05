def read_file(file_path: str) -> list:
    """
    Read a file and return its lines.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read().splitlines()
    except IOError as e:
        print(f'Error: Could not read file {file_path}: {e}')
        return []

