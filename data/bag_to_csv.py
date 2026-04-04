import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import pandas as pd


def bag_to_csv(bag_path, topic_name, output_csv):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', '')
    )

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    msg_type = get_message(type_map[topic_name])

    data_list = []
    expected_len = None

    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic != topic_name:
            continue

        msg = deserialize_message(data, msg_type)

        if expected_len is None:
            expected_len = len(msg.data)
            print("Số chiều:", expected_len)

        if len(msg.data) != expected_len:
            continue

        row = [t] + list(msg.data)
        data_list.append(row)

    # tạo column
    columns = ["timestamp"] + [f"data_{i}" for i in range(expected_len)]

    df = pd.DataFrame(data_list, columns=columns)
    df.to_csv(output_csv, index=False)

    print("Saved:", output_csv)


if __name__ == "__main__":
    bag_to_csv(
        "/home/wicom/ros2_ws/data/training_bag_1703",
        "/uam/debug_state",
        "1703_clean.csv"
    )