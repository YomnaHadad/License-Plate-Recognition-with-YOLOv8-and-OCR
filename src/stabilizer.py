from collections import defaultdict, deque

# Plate stabilization for video
plate_history = defaultdict(lambda: deque(maxlen=10))   
plate_final = {}

def get_box_id(x1, y1, x2, y2):
  # use rounded coordinates as a pseudo id
    return f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"

def get_stable_plate(box_id, new_text):

    if new_text:
        plate_history[box_id].append(new_text)

        most_common = max(
            set(plate_history[box_id]),
            key=plate_history[box_id].count
        )

        plate_final[box_id] = most_common

    return plate_final.get(box_id, "")