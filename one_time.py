import re

def change_commas_to_semicolons_movie_list(file_path:str):
    with open(file_path, "r") as file:
        text = file.readline()
    if ";" in text:
        return 0
    else:
        with open(file_path, "r") as file:
            text_lines = file.readlines()
        for line_number, line in enumerate(text_lines):
            replaced_text = re.sub(r"(\d+)(?:,)(\d+)(?:,)", "\g<1>;\g<2>;", line)
            text_lines[line_number] = replaced_text
        with open(file_path, "w") as file:
            file.writelines(text_lines)