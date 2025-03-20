import numpy as np

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    return 0

def feature_search_demo(data):
    for i in range(1, len(data[0])):
        print(f"On the {i}th level of the search tree")

data = np.loadtxt("CS170_Small_Data__96.txt")

feature_search_demo(data)