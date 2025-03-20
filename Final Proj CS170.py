import numpy as np

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    new_data = np.copy(data)

    for i in range(len(new_data)):
        for j in range(1, len(new_data[0])):
            if j not in current_set and j != feature_to_add:
                new_data[i, j] = 0

    number_correctly_classified = 0
    for i in range(len(new_data)):
        object_to_classify = new_data[i, 1:]
        label_object_to_classify = new_data[i,0]
        
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for k in range(len(new_data)):
            if k != i:
                distance = np.sqrt(np.sum((object_to_classify - new_data[k, 1:]) ** 2))

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = new_data[nearest_neighbor_location, 0]

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified = number_correctly_classified + 1

    return number_correctly_classified / len(new_data)
    

def feature_search_demo(data):
    print(f"This dataset has {len(data[0]) - 1} features and {len(data)} instances")

    total_accuracy = leave_one_out_cross_validation(data, list(range(1, len(data[0]) - 1)), None)

    print(f"Running nearest neighbor with {len(data[0]) - 1} using leave one out, the resulting accuracy is: {total_accuracy}")

    print("Beginning Search:")

    current_set_of_features = []
    for i in range(1, len(data[0])):
        print(f"On the {i}th level of the search tree")

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

                print(f"Using feature(s) {clean_format} accuracy is {accuracy:.2f}")

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k

        if feature_to_add_at_this_level is not None:
            current_set_of_features.append(feature_to_add_at_this_level)

            features = []

            for feature in current_set_of_features:
                features.append(str(feature))

            clean_format = "{" + ", ".join(features) + "}"

            print(f"Feature set {clean_format} was best, accuracy is {best_so_far_accuracy:.2f}")

        print(f"On level {i}, I added {feature_to_add_at_this_level} to the current set")

data = np.loadtxt("CS170_Small_Data__96.txt")

feature_search_demo(data)
