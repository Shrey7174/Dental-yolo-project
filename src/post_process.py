
FDI_INFO = {
    0: {'fdi': 13, 'jaw': 'upper'}, 1: {'fdi': 23, 'jaw': 'upper'},
    2: {'fdi': 33, 'jaw': 'lower'}, 3: {'fdi': 43, 'jaw': 'lower'},
    4: {'fdi': 21, 'jaw': 'upper'}, 5: {'fdi': 41, 'jaw': 'lower'},
    6: {'fdi': 31, 'jaw': 'lower'}, 7: {'fdi': 11, 'jaw': 'upper'},
    8: {'fdi': 16, 'jaw': 'upper'}, 9: {'fdi': 26, 'jaw': 'upper'},
    10: {'fdi': 36, 'jaw': 'lower'}, 11: {'fdi': 46, 'jaw': 'lower'},
    12: {'fdi': 14, 'jaw': 'upper'}, 13: {'fdi': 34, 'jaw': 'lower'},
    14: {'fdi': 44, 'jaw': 'lower'}, 15: {'fdi': 24, 'jaw': 'upper'},
    16: {'fdi': 22, 'jaw': 'upper'}, 17: {'fdi': 32, 'jaw': 'lower'},
    18: {'fdi': 42, 'jaw': 'lower'}, 19: {'fdi': 12, 'jaw': 'upper'},
    20: {'fdi': 17, 'jaw': 'upper'}, 21: {'fdi': 27, 'jaw': 'upper'},
    22: {'fdi': 37, 'jaw': 'lower'}, 23: {'fdi': 47, 'jaw': 'lower'},
    24: {'fdi': 15, 'jaw': 'upper'}, 25: {'fdi': 25, 'jaw': 'upper'},
    26: {'fdi': 35, 'jaw': 'lower'}, 27: {'fdi': 45, 'jaw': 'lower'},
    28: {'fdi': 18, 'jaw': 'upper'}, 29: {'fdi': 28, 'jaw': 'upper'},
    30: {'fdi': 38, 'jaw': 'lower'}, 31: {'fdi': 48, 'jaw': 'lower'},
}

def apply_anatomical_rules(yolo_boxes, image_height, class_names):
    """Filters YOLO predictions based on anatomical rules."""
    vertical_midline = image_height / 2
    valid_boxes = []

    for box in yolo_boxes:
        class_id = int(box.cls[0])
        box_coords = box.xyxy[0]
        box_center_y = (box_coords[1] + box_coords[3]) / 2

        expected_jaw = FDI_INFO.get(class_id, {}).get('jaw')
        detected_jaw = 'upper' if box_center_y < vertical_midline else 'lower'

        if expected_jaw == detected_jaw:
            valid_boxes.append(box)
        else:
            tooth_name = class_names.get(class_id, 'Unknown')
            print(f"Post-processing: Removed '{tooth_name}' predicted in the wrong jaw ({detected_jaw}).")

    return valid_boxes