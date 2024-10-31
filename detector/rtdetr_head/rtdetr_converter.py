import torch
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib


def to_feats_list(feats):
    # 如果输入是字典，并且字典中包含以'stage'开头的键
    if isinstance(feats, dict) and any(key.startswith('stage') for key in feats.keys()):
        # 将字典中以 'stage' 开头的键按照数字顺序排序并取最后三个值
        sorted_keys = sorted([key for key in feats.keys() if key.startswith('stage')], 
                             key=lambda x: int(x.replace('stage', '')))
        feats_list = [feats[key] for key in sorted_keys[-3:]]
    elif isinstance(feats, list):
        # 如果输入已经是列表，则取列表的最后三个元素
        feats_list = feats[-3:]
    else:
        # 如果输入既不是字典也不是列表，抛出异常
        raise ValueError("Input must be either a list or a dictionary with keys starting with 'stage'.")
    
    return feats_list



# def to_feats_list(feats): # dict{2: tensor, 3: tensor, 4: tensor}
#     # 如果输入是字典，并且所有的键都是整数，转换为按键排序的列表
#     if isinstance(feats, dict) and all(isinstance(key, int) for key in feats.keys()):
#         # 将键按升序排序并从字典中提取特征
#         feats_list = [feats[key] for key in sorted(feats.keys())]
#     elif isinstance(feats, list):
#         # 如果输入已经是列表，直接返回
#         return feats
#     else:
#         # 如果输入既不是字典也不是列表，或者字典的键不全是整数，抛出异常
#         raise ValueError("Input must be either a list or a dictionary with integer keys.")
    
#     return feats_list


def move_to_device(container, device):
    if isinstance(container, torch.Tensor):
            return container.to(device)
    elif isinstance(container, dict):
            return {k: move_to_device(v, device) for k, v in container.items()}
    elif isinstance(container, list):
            return [move_to_device(v, device) for v in container]
    else:
            return container


def per_yolo_to_target_dict(yolo_bbox, img_width, img_height, image_id):
    """
    Convert a batch of YOLO formatted bounding boxes to the format shown in the image.
    :param yolo_bbox: Tensor containing (class_id, cx, cy, nw, nh) normalized
    :param img_width: Width of the image
    :param img_height: Height of the image
    :param image_id: The ID of the image
    :return: Dictionary with target information
    """
    box_w = yolo_bbox[:, 3] * img_width
    box_h = yolo_bbox[:, 4] * img_height

    # print((torch.tensor([img_width, img_height], dtype=torch.int32)).shape)
    # Format similar to the image provided
    target_dict = {
        'boxes': yolo_bbox[:, 1:].clone().detach().to(dtype=torch.float32),
        'labels': yolo_bbox[:, 0].clone().detach().long(),
        'image_id': torch.tensor([image_id]).long(),
        'area': (box_w * box_h).clone().detach().to(dtype=torch.float32),
        'iscrowd': torch.zeros(yolo_bbox.shape[0]).long(),
        'orig_size': torch.tensor([img_width, img_height]).long(),
        'size': torch.tensor([img_width, img_height]).long()
    }
    
    return target_dict

def convert_yolo_batch_to_targets_format(batch_bboxes, img_width, img_height, labelmap=("car", "pedestrian")):
    """
    Convert a batch of YOLO bboxes to the target format.
    :param batch_bboxes: Tensor of shape (batch_size, num_boxes, 5) containing YOLO bboxes
    :param img_width: Width of the image
    :param img_height: Height of the image
    :param labelmap: Tuple containing the label names
    :return: List of dictionaries with target information
    """
    # Check if batch_bboxes is already a dictionary with the expected keys
    expected_keys = {'boxes', 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'size'}
    if isinstance(batch_bboxes, dict) and expected_keys <= batch_bboxes.keys():
        return batch_bboxes
    
    batch_size = batch_bboxes.shape[0]
    converted_data = []
    
    for i in range(batch_size):
        image_id = i + 1  # Assuming image IDs start from 1
        image_bboxes = batch_bboxes[i]
        non_zero_boxes = image_bboxes[image_bboxes[:, 3] * image_bboxes[:, 4] > 0]  # Filter out all-zero bboxes
        # print("non_zero_boxes: ", non_zero_boxes)
        # print("non_zero_boxes.size(0): ", non_zero_boxes.size(0))
        
        if non_zero_boxes.size(0) > 0:
            # for bbox in non_zero_boxes:
            converted_data.append(per_yolo_to_target_dict(non_zero_boxes, img_width, img_height, image_id))
            # print("converted_data: ", converted_data)
        else:
            # Handle the case for an image with no annotations
            converted_data.append({
                'boxes': torch.empty((0, 4), dtype=torch.float32),
                'labels': torch.empty((0,)).long(),
                'image_id': torch.tensor([image_id]).long(),
                'area': torch.empty((0,), dtype=torch.float32),
                'iscrowd': torch.empty((0,)).long(),
                'orig_size': torch.tensor([img_width, img_height]).long(),
                'size': torch.tensor([img_width, img_height]).long()
            })
    
    return converted_data # List of dict


# * from yolo_batch(normalized cxcywh) to coco_format data
def yolo_to_coco(yolo_bbox, img_width, img_height):
    """
    Convert a single YOLO bbox to COCO format.
    :param yolo_bbox: Tensor containing (class_id, cx, cy, w, h) normalized
    :param img_width: Width of the image
    :param img_height: Height of the image
    :return: Dictionary in COCO format
    """
    class_id, cx, cy, w, h = yolo_bbox
    x = (cx - w / 2) * img_width
    y = (cy - h / 2) * img_height
    width = w * img_width
    height = h * img_height

    return {
        "category_id": int(class_id.item()) + 1,  # from 0,1 to 1,2;  0 is background
        "bbox": [x.item(), y.item(), width.item(), height.item()],
        "area": width.item() * height.item(),
        "iscrowd": 0
    }

def convert_yolo_batch_to_coco_format(bboxes, img_width=240, img_height=240, labelmap=("car", "pedestrian")):
    # Check if batch_bboxes is already a dictionary with the expected keys
    expected_keys = {'images', 'annotations', 'categories', 'info', 'licenses', 'type'}
    if isinstance(bboxes, dict) and expected_keys <= bboxes.keys():
        return bboxes
    
    batch_size = bboxes.shape[0]
    coco_annotations = []
    coco_images = []
    coco_categories = [{"id": id + 1, "name": name, "supercategory": "none"} for id, name in enumerate(labelmap)]

    for i in range(batch_size):
        image_id = i + 1
        valid_bboxes = [bbox for bbox in bboxes[i] if bbox[3] > 0 and bbox[4] > 0]

        coco_images.append({
            "id": image_id,
            "file_name": "n.a", #f"image_{i+1}.jpg",
            "width": img_width,
            "height": img_height
        })

        if not valid_bboxes:
            # Add an annotation entry with no bbox but with an image_id to denote an image without annotations.
            coco_annotations.append({
                "id": len(coco_annotations) + 1,
                "image_id": image_id,
                "category_id": 0,  # Assuming 0 or another identifier could signify no objects
                "bbox": [],
                "area": 0,
                "iscrowd": 0
            })
        else:
            for j, bbox in enumerate(valid_bboxes):
                annotation = yolo_to_coco(bbox.clone().detach(), img_width, img_height)
                annotation.update({"image_id": image_id, "id": len(coco_annotations) + 1})
                coco_annotations.append(annotation)

    coco_data = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
        "info": {},
        "licenses": [],
        "type": "instances"
    }
    
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        coco_gt = COCO()
        coco_gt.dataset = coco_data
        coco_gt.createIndex()
        # coco_pred = coco_gt.loadRes(results)

        # coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')

    return coco_gt




def to_coco_format(gts, detections, categories, height=240, width=304):
    """
    utilitary function producing our data in a COCO usable format
    """
    annotations = []
    results = []
    images = []

    # to dictionary
    for image_id, (gt, pred) in enumerate(zip(gts, detections)):
        im_id = image_id + 1

        images.append(
            {"date_captured": "2019",
             "file_name": "n.a",
             "id": im_id,
             "license": 1,
             "url": "",
             "height": height,
             "width": width})

        for bbox_gt in gt:
            x1, y1 = bbox_gt['x'], bbox_gt['y']
            w, h = bbox_gt['w'], bbox_gt['h']
            area = w * h

            annotation = {
                "area": float(area),
                "iscrowd": False,
                "image_id": im_id,
                "bbox": [x1, y1, w, h],
                "category_id": int(bbox_gt['class_id']) + 1,
                "id": len(annotations) + 1
            }
            annotations.append(annotation)

        for bbox_dt in pred:
            image_result = {
                'image_id': im_id,
                'category_id': int(bbox_dt['class_id']) + 1,
                'score': float(bbox_dt['class_confidence']),
                'bbox': [bbox_dt['x'], bbox_dt['y'], bbox_dt['w'], bbox_dt['h']],
            }
            results.append(image_result)

    dataset = {"info": {},
               "licenses": [],
               "type": 'instances',
               "images": images,
               "annotations": annotations,
               "categories": categories}
    
    return dataset, results

def _coco_eval(gts, detections, height, width, labelmap=("car", "pedestrian"), return_aps: bool = True):
    """simple helper function wrapping around COCO's Python API
    :params:  gts iterable of numpy boxes for the ground truth
    :params:  detections iterable of numpy boxes for the detections
    :params:  height int
    :params:  width int
    :params:  labelmap iterable of class labels
    """
    categories = [{"id": id + 1, "name": class_name, "supercategory": "none"}
                  for id, class_name in enumerate(labelmap)]

    num_detections = 0
    for detection in detections:
        num_detections += detection.size

    # Meaning: https://cocodataset.org/#detection-eval
    # out_keys = ('AP', 'AP_50', 'AP_75', 'AP_S', 'AP_M', 'AP_L')
    out_keys = ('AP', 'AP_50', 'AP_75', 'AP_S', 'AP_M', 'AP_L', 'AR_1', 'AR_10', 'AR_100', 'AR_S', 'AR_M', 'AR_L')
    out_dict = {k: 0.0 for k in out_keys}

    if num_detections == 0:
        # Corner case at the very beginning of the training.
        # print('no detections for evaluation found.')
        return out_dict if return_aps else None

    dataset, results = to_coco_format(gts, detections, categories, height=height, width=width)

    # coco_gt = COCO()
    # coco_gt.dataset = dataset
    # coco_gt.createIndex()
    # coco_pred = coco_gt.loadRes(results)

    # coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    # torch.save(coco_eval, "/home/catlab/py_code/gt4dvs/save_dir/processing/coco_eval.pt")
    # # coco_eval.params.iouThrs = np.linspace(0.2, 0.95, int(np.round((0.95 - 0.2) / 0.05)) + 1, endpoint=True)
    # coco_eval.params.imgIds = np.arange(1, len(gts) + 1, dtype=int)
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    if return_aps:
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            coco_gt = COCO()
            coco_gt.dataset = dataset
            coco_gt.createIndex()
            coco_pred = coco_gt.loadRes(results)

            coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
            torch.save(coco_eval, "/home/catlab/py_code/gt4dvs/save_dir/processing/coco_eval.pt")
            # coco_eval.params.iouThrs = np.linspace(0.2, 0.95, int(np.round((0.95 - 0.2) / 0.05)) + 1, endpoint=True)
            coco_eval.params.imgIds = np.arange(1, len(gts) + 1, dtype=int)
            coco_eval.evaluate()
            coco_eval.accumulate()
            # info: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
            coco_eval.summarize()
        for idx, key in enumerate(out_keys):
            out_dict[key] = coco_eval.stats[idx]
        return out_dict


# Example usage
# Assume bboxes is a tensor with shape (batch_size, num_boxes, 5) with YOLO format
# bboxes = torch.rand((2, 8, 5))  # Example tensor, replace with your own data
# bboxes = torch.tensor([[[1.0000, 0.1578, 0.4583, 0.2094, 0.1167],
#          [1.0000, 0.1641, 0.4583, 0.0469, 0.1333],
#          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#          [1.0000, 0.3656, 0.5000, 0.0875, 0.1750],
#          ],

#          [[1.0000, 0.1578, 0.4583, 0.2094, 0.1167],
#          [0.0000, 0.1641, 0.4583, 0.0469, 0.1333],
#          [0.0000, 0.3453, 0.4542, 0.1531, 0.0917],
#          [1.0000, 0.3656, 0.5000, 0.0875, 0.1750],
#          ],

#         [[0.0000, 0.1578, 0.4583, 0.2094, 0.1167],
#          [1.0000, 0.1641, 0.4583, 0.0469, 0.1333],
#          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#          ],

#         [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
# ])
# targets_format_data = convert_yolo_batch_to_targets_format(bboxes, img_width=304, img_height=240)
# # print(targets_format_data[3])

