from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils
from object_detection.utils import ops as utils_ops

PATH_TO_LABELS = './models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)