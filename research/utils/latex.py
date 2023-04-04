from typing import Dict, List, Optional

import numpy as np


def color_entry(entry: str, value: float, color_map: Dict[float, str]) -> str:
    """Color an entry in the table based on the value and the color map.

    Args:
        entry (str): The entry to color.
        value (float): The value of the entry.
        color_map (Dict[float, str]): The color map.

    Returns:
        str: The colored entry.
    """
    for key in sorted(color_map.keys()):
        if value <= key:
            return f"\\cellcolor[rgb]{{{color_map[key]}}} {entry}"
    return entry


def color_text(entry: str, value: Optional[float], color_map: Dict[float, str]) -> str:
    """Color an raw latext text to the desired color.

    Args:
        entry (str): The text to color.
        value (Optional[float]): The value of the entry.
        color_map (Dict[float, str]): The color map.

    Returns:
        str: The colored text.
    """
    for key in sorted(color_map.keys()):
        if value <= key:
            return f"\\textcolor[rgb]{{{color_map[key]}}}{{{entry}}}"
    return entry


def _check_if_num_and_get_value(entry: str) -> Optional[float]:
    """Check if the entry is a number and if so, return the value.

    Args:
        entry (str): The entry to check.

    Returns:
        Optional[float]: The value of the entry.
    """

    if entry.replace(".", "", 1).isdigit():
        return float(entry)
    elif "\\pm" in entry and entry.split("\\pm")[0].replace(".", "", 1).isdigit():
        # check if \pm is in the entry
        return float(entry.split("\\pm")[0])
    else:
        return None


def create_color_map(
    start_color: str, end_color: str, start: float, end: float, n_colors: int
) -> Dict[float, str]:
    """Create a color map.

    Args:
        start_color (str): The start color.
        end_color (str): The end color.
        start (float): The start value.
        end (float): The end value.
        n_colors (int): The number of colors.

    Returns:
        Dict[float, str]: The color map.
    """

    start_rgb = start_color.split(",")
    end_rgb = end_color.split(",")
    color_map = {}

    values = np.linspace(start, end, n_colors)

    for i in range(n_colors):
        rgb = [
            float(start_rgb[j])
            + (float(end_rgb[j]) - float(start_rgb[j])) * i / n_colors
            for j in range(3)
        ]
        color_map[values[i]] = f"{rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f}"
    return color_map


def generate_latex_table(
    data: List[Optional[List[str]]],
    caption: str = "TODO: Caption",
    label: str = "TODO-label",
    color_map: Optional[Dict[float, str]] = None,
    small: bool = False,
    table_extras: str = "",
) -> str:
    """Generate a latex table.

    Args:
        data (List[Optional[List[str]]]): The data to put in the table. None values are converted to `\midrule`s.
        caption (str, optional): The caption of the table.
        label (str, optional): The label of the table.
        color_map (Optional[Dict[float, str]], optional): The color map to color cells by
        small (bool): Make the table have small text

    Returns:
        str: The latex table as a string. Can be printed or written to a file and included in a latex document.
    """
    assert (
        len(data) > 1
    ), "data must contain at least one row of header and one row of data"
    assert all(
        len(row) == len(data[0]) for row in data if row is not None
    ), "all rows in data must have the same number of columns"
    n_rows = len(data)
    n_cols = len(data[0])
    column_format = " ".join(["l"] * n_cols)

    max_char_length = []  # the maximum number of characters in each column
    for j in range(n_cols):
        max_length = 0
        for i in range(1, n_rows):
            if data[i] is None:
                continue
            if data[i][j] is not None:
                max_length = max(max_length, len(str(data[i][j])))
        if color_map is None:
            max_char_length.append(max_length)
        else:
            # if a color map is given, we need to add extra padding for the color
            max_char_length.append(max_length + 34)

    # treat the first row as the header
    header = " & ".join(
        [str(col).rjust(max_char_length[j]) for j, col in enumerate(data[0])]
    )

    # treat the rest of the rows as the data
    rows = []
    for i in range(1, n_rows):
        if data[i] is None:
            rows.append("    \\midrule")
        else:
            entries = []
            # iterate over the columns and color the entries if a color map is given
            for j in range(n_cols):
                entry = str(data[i][j])
                value = _check_if_num_and_get_value(entry)
                # check if value is a number and if so, color it
                if color_map is not None and value is not None:
                    entries.append(
                        # color_entry(entry, value, color_map).rjust(max_char_length[j])
                        color_text(entry, value, color_map).rjust(max_char_length[j])
                    )
                else:
                    entries.append(entry.rjust(max_char_length[j]))
            row = " & ".join(entries) + " \\\\"
            rows.append(row)
    table = "\n".join(rows)

    if small:
        latex = f"""\\begin{{table*}}
  \\small
  \\caption{{{caption}}}
  \\label{{{label}}}
  \\centering
  \\begin{{tabular}}{{{column_format}}}
    \\toprule
{header} \\\\
    \\midrule
{table}
    \\bottomrule
  {table_extras}
  \\end{{tabular}}
\\end{{table*}}
"""
    else:
        latex = f"""\\begin{{table*}}
  \\caption{{{caption}}}
  \\label{{{label}}}
  \\centering
  \\begin{{tabular}}{{{column_format}}}
    \\toprule
{header} \\\\
    \\midrule
{table}
    \\bottomrule
  {table_extras}
  \\end{{tabular}}
\\end{{table*}}
"""
    return latex


if __name__ == "__main__":
    data = [
        ["Dataset", "Task", "MLP-C", "MTM", "MTM-SM", "MTM-FT"],
        ["Expert", "BC", "108.4\pm0.3", "0.0\pm0.0", "0000", "5"],
        ["Expert", "RCBC", "108.5\pm0.3", "0000000", "0000000", "10"],
        ["Expert", "ID", "0.0022\pm0.0005", "0000000", "0000000", "15"],
        ["Expert", "FD", "0.12\pm0.015", "0000000", "0000000", "20"],
        None,
        None,
        ["Medium-Expert", "BC", "106.7\pm2.5", "0000000", "0000", "30"],
        ["Medium-Expert", "RCBC", "109.1\pm0.8", "0000000", "80.0\pm0.0", "40"],
        None,
        ["Medium-Expert", "ID", "0.0036\pm0.0012", "0000000", "70.0\pm0.0", "60"],
    ]
    caption = "The caption."
    label = "d4rl-walker2d-table"

    # white to light blue color map
    color_map = create_color_map("1.00,1.00,1.00", "0.4,0.75,0.90", -1, 120, 50)

    print(generate_latex_table(data, caption, label, None))
    # print(generate_latex_table(data, caption, label, color_map))
