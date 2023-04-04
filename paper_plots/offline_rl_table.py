from research.utils.latex import create_color_map, generate_latex_table

# data taken from https://fairwandb.org/mtm_team/d4rl_mtm_cont

# double check numbers
# add average
data_str = """[
    ["Environment", "Dataset"      , "BC"  , "CQL"  , "IQL"  , "TT"   , "MOPO", "RsV"  , "DT"    , "\\\\textbf{MTM (Ours)}"],
    ["HalfCheetah", "Medium-Replay", "36.6", "45.5" , "44.2" , "41.9" , "42.3", "38.0" , "36.6"  , "42.0"],
    ["Hopper"     , "Medium-Replay", "18.1", "95.0" , "94.7" , "91.5" , "28.0", "73.5" , "82.7"  , "92.3"],
    ["Walker2d"   , "Medium-Replay", "26.0", "77.2" , "73.9" , "82.6" , "17.8", "60.6" , "66.6"  , "78.3"],
    ["HalfCheetah", "Medium"       , "42.6", "44.0" , "47.4" , "46.9" , "53.1", "41.6" , "42.0"  , "43.3"],
    ["Hopper"     , "Medium"       , "52.9", "58.5" , "66.3" , "61.1" , "67.5", "60.2" , "67.6"  , "64.1"],
    ["Walker2d"   , "Medium"       , "75.3", "72.5" , "78.3" , "79.0" , "39.0", "71.7" , "74.0"  , "77.0"],
    ["HalfCheetah", "Medium-Expert", "55.2", "91.6" , "86.7" , "95.0" , "63.7", "92.2" , "86.8"  , "91.6"],
    ["Hopper"     , "Medium-Expert", "52.5", "105.4", "91.5" , "110.0", "23.7", "101.7", "107.6", "110.9"],
    ["Walker2d"   , "Medium-Expert", "107.5","108.8", "109.6", "101.9", "44.6", "106.0", "108.1", "109.5"],
]"""

"""

"TT"   ,
"41.9" ,
"91.5" ,
"82.6" ,
"46.9" ,
"61.1" ,
"79.0" ,
"95.0" ,
"110.0",
"101.9",


"""


if __name__ == "__main__":
    caption = "The caption."
    label = "d4rl-offline-rl-results"
    data = eval(data_str)

    # add average row
    number_data = [x[2:] for x in data[1:]]
    number_data = [[float(y) for y in x] for x in number_data]
    average_data = [sum(x) / len(x) for x in zip(*number_data)]

    # add average row, trncate string to 1 decimal value
    data.append(
        [
            "Average",
            "",
        ]
        + [f"{x:.1f}" for x in average_data]
    )

    # white to light green color map
    # green = "0.4,0.75,0.4"
    # color_map = create_color_map("1.00,1.00,1.00", "0.4,0.75,0.4", -1, 120, 100)
    # blue
    color_map = create_color_map("1.00,1.00,1.00", "0.4,0.75,0.90", 11, 111, 100)
    print(generate_latex_table(data, caption, label, None))
    print(generate_latex_table(data, caption, label, color_map=color_map))
