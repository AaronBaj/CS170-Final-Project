import numpy as np

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    return 1

def feature_search_demo(data):
    current_set_of_features = []

    for i in range(1, len(data[0])):
        print(f"On the {i}th level of the search tree")
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0
        for k in range(1, len(data[0])):
            if k not in current_set_of_features:
                print(f"--Consider adding the {k} feature")
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k + 1)

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k

        print(f"On level {i}, I added {feature_to_add_at_this_level} to the current set")

data = np.loadtxt("CS170_Small_Data__96.txt")

feature_search_demo(data)
