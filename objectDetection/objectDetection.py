#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2


def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, classes, COLORS, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    draw_text(img, 0, label, x, y, color)
    draw_text(img, 1, str(round(confidence * 100, 2)) + ' %', x, y, color)


def draw_text(img, line, text, x, y, color):
    text_color = [0, 0, 0]
    bg_color = color
    x_offset = 5
    y_offset = 15
    x = x + x_offset
    y = y + y_offset
    box_y_offset = -10
    box_padding = 5
    line_height = 12
    thickness = 1
    line_offset = line * (line_height + 2 * box_padding)

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
    cv2.rectangle(img, (x - box_padding, y + box_y_offset - box_padding + line_offset),
                  (x + text_size[0] + (2 * box_padding), y + box_y_offset + box_padding + text_size[1] + line_offset),
                  bg_color, -1)
    cv2.putText(img, text, (x, y + line_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness)
