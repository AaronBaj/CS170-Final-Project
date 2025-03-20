import numpy as np
import time

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    
    features_to_use = list(current_set)

    if feature_to_add is not None:
        features_to_use.append(feature_to_add)

    number_correctly_classified = 0
    for i in range(len(data)):
        object_to_classify = data[i, features_to_use]
        label_object_to_classify = data[i,0]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for k in range(len(data)):
            if k != i:
                distance = np.sqrt(np.sum((object_to_classify - data[k, features_to_use]) ** 2))

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location, 0]

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified = number_correctly_classified + 1

    return number_correctly_classified / len(data)
    

def feature_search_demo(data):
    time_start = time.time()

    print(f"This dataset has {len(data[0]) - 1} features (not including the class feature) and {len(data)} instances")

    total_accuracy = leave_one_out_cross_validation(data, list(range(1, len(data[0]) - 1)), None)

    print(f"Running nearest neighbor with {len(data[0]) - 1} features using leaveing-one-out, the resulting accuracy is: {total_accuracy * 100:.1f}%")

    print("Beginning Search:")
    best_set = []
    current_set_of_features = []

    for i in range(1, len(data[0])):
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0

        for k in range(1, len(data[0])):
            if k not in current_set_of_features:
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)

                features = []

                for feature in current_set_of_features:
                    features.append(str(feature))

                features.append(str(k))

                clean_format = "{" + ", ".join(features) + "}"

                print(f"    Using feature(s) {clean_format} accuracy is {accuracy * 100:.1f}%")

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k

        if feature_to_add_at_this_level is not None:
            current_set_of_features.append(feature_to_add_at_this_level)

            if best_so_far_accuracy > total_accuracy:
                total_accuracy = best_so_far_accuracy
                best_set = list(current_set_of_features) 

            features = []

            for feature in current_set_of_features:
                features.append(str(feature))

            clean_format = "{" + ", ".join(features) + "}"

            print(f"Feature set {clean_format} was best, accuracy is {best_so_far_accuracy * 100:.1f}%")
    time_end = time.time()
    
    features = []

    for feature in best_set:
        features.append(str(feature))

    clean_format_new = "{" + ", ".join(features) + "}"

    print(f"\nSearch finished, the best feature set is {clean_format_new}, with an accuracy of {total_accuracy * 100:.1f}%")
    print(f"Total runtime: {time_end - time_start:.1f} seconds")
        

def backward_elimination(data):
    time_start = time.time()

    print(f"This dataset has {len(data[0]) - 1} features (not including the class feature) and {len(data)} instances")
    current_set_of_features = list(range(1, len(data[0])))

    total_accuracy = leave_one_out_cross_validation(data, current_set_of_features, None)

    best_total_accuracy = total_accuracy

    best_set = list(current_set_of_features)

    print(f"Running nearest neighbor with {len(data[0]) - 1} using leave one out, the resulting accuracy is: {total_accuracy * 100:.1f}%")

    print("Beginning Search:")
    

    for i in range(len(data[0]) - 2):
        feature_to_remove_at_this_level = None
        best_so_far_accuracy = 0

        for k in current_set_of_features:
            tmp_features = list(current_set_of_features)
            tmp_features.remove(k)

            accuracy = leave_one_out_cross_validation(data, tmp_features, None)

            features = [str(feature) for feature in tmp_features]
            clean_format = "{" + ", ".join(features) + "}"
            print(f"    Using feature(s) {clean_format} accuracy is {accuracy * 100:.1f}%")

            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_remove_at_this_level = k

        if feature_to_remove_at_this_level is not None:
            current_set_of_features.remove(feature_to_remove_at_this_level)

            if best_so_far_accuracy > best_total_accuracy:
                best_total_accuracy = best_so_far_accuracy
                best_set = list(current_set_of_features) 

            features = []

            for feature in current_set_of_features:
                features.append(str(feature))

            clean_format = "{" + ", ".join(features) + "}"

            print(f"Feature set {clean_format} was best, accuracy is {best_so_far_accuracy * 100:.1f}%")
    time_end = time.time()
    
    features = []

    for feature in best_set:
        features.append(str(feature))

    clean_format_new = "{" + ", ".join(features) + "}"

    print(f"\nSearch finished, the best feature set is {clean_format_new}, with an accuracy of {best_total_accuracy * 100:.1f}%")
    print(f"Total runtime: {time_end - time_start:.1f} seconds")

choice_data = input("Please type 1 for Small Dataset or 2 for Large dataset: ")

choice = input("Please type 1 for Forward Selection or 2 for Backward Elimination: ")


if choice_data == '1':
    data = np.loadtxt("CS170_Small_Data__96.txt")
    print("Small dataset loaded.")
elif choice_data == '2':
    data = np.loadtxt("CS170_Large_Data__40.txt")
    print("Large dataset loaded.")
else:
    print("Invalid dataset selection, run code again.")

if choice == '1':
    print("Running Forward Selection:")
    feature_search_demo(data)
elif choice == '2':
    print("Running Backward Elimination:")
    backward_elimination(data)
else:
    print("Invalid selection, run code again")

