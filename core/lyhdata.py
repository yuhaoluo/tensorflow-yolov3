import tensorflow as tf
import core.utils as utils
from core.config import cfg
import numpy as np
import os
import cv2
import random
class Dataset(object):
    def __init__(self,dataset_type,num_parallel=4):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG
        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train'else cfg.TEST.INPUT_SIZE
        #self.train_input_size = 416
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = 3
        self.max_bbox_per_scale = 150
        self.nrof_images = len(open(self.annot_path, 'r').readlines()) #* 1000
        self.batches_per_epoch = int(np.floor(self.nrof_images/self.batch_size))
        self.num_parallel = num_parallel

        self.next_element = self.produce_data_iter()

    #def set_train_input_size(self,size):
    #    self.train_input_size = size

    ## use in TEST 
    # def load_annotations(self, dataset_type):
    #     ## return
    #     ## type: list, etc[ [img_path xmin1,ymin1,xmax1,ymax1,c1 xmin2,ymin2,xmax2,ymax2,c2 ],...]
    #     annot_path = '/home/luoyuhao/Tools/yolov3/tensorflow-yolov3/data/images/train.txt'
    #     with open(annot_path, 'r') as f:
    #         txt = f.readlines()
    #         annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    #         # 处理换行符
    #     #np.random.shuffle(annotations)
    #     return annotations  
    def get_dataset(self):
        self.train_input_size = random.choice(self.train_input_sizes)

        dataset = tf.data.TextLineDataset(self.annot_path)
        dataset = dataset.shuffle(self.nrof_images)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.batch_size*2)
        dataset = dataset.map(
            lambda x: tf.py_func(self.get_batch_data,
                                inp=[x],
                                Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,tf.float32,tf.float32]),
            num_parallel_calls = self.num_parallel
        )
        return dataset

    def produce_data_iter(self):
        ## TEST
        #path_list = self.load_annotations('')
        #path_list = path_list * 1000
        #train_data = tf.data.Dataset.from_tensor_slices(path_list)
        self.train_input_size = random.choice(self.train_input_sizes)

        train_data = tf.data.TextLineDataset(self.annot_path)
        train_data = train_data.shuffle(self.nrof_images)
        train_data = train_data.batch(self.batch_size)
        #train_data = train_data.prefetch(self.batch_size*2)
        train_data = train_data.map(
            lambda x: tf.py_func(self.get_batch_data,
                                inp=[x],
                                Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,tf.float32,tf.float32]),
            num_parallel_calls = self.num_parallel
        )
        
        iterator = train_data.make_initializable_iterator()
        next_element = iterator.get_next()
        self.train_data_init_op = iterator.initializer
        return next_element

# def load_annotations(self,annotations):
#     annotations = [line.strip() for line in annotations if len(line.strip().split()[1:]) != 0]
#     return annotations

    def parse_annotation(self,annotation):
        if 'str' not in str(type(annotation)):
            annotation = annotation.decode()
        line = annotation.split()
        image_path = line[0]
        #print(image_path)
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image = np.array(cv2.imread(image_path))
        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        return image, bboxes

    def bbox_iou(self,boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self,bboxes):
        
        # per_scale_label [out_size,out_size,3,c+5]
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                            5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]
            ## onehot标签，同时做label smooth处理
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                #在每一个scale确定当前box由哪个cell的anchors确定，并且iou大于0.3才最终匹配
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def get_batch_data(self,batch_annotations):
        self.train_output_sizes = self.train_input_size // self.strides
        batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3),dtype=np.float32)

        batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                        self.anchor_per_scale, 5 + self.num_classes),dtype=np.float32)
        batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                        self.anchor_per_scale, 5 + self.num_classes),dtype=np.float32)
        batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                        self.anchor_per_scale, 5 + self.num_classes),dtype=np.float32)

        batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4),dtype=np.float32)
        batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4),dtype=np.float32)
        batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4),dtype=np.float32)
        num = 0
        for annotation in batch_annotations:
            image,bboxes = self.parse_annotation(annotation)
            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
            batch_label_sbbox[num,:,:,:,:] = label_sbbox
            batch_label_mbbox[num,:,:,:,:] = label_mbbox
            batch_label_lbbox[num,:,:,:,:] = label_lbbox
            batch_sbboxes[num,:,:] = sbboxes
            batch_mbboxes[num,:,:] = mbboxes
            batch_lbboxes[num,:,:] = lbboxes   
            batch_image[num,:,:,:] = image
            num += 1    
            #print(mbboxes[0:3,:])
        return batch_image,batch_label_sbbox,batch_label_mbbox,batch_label_lbbox,\
            batch_sbboxes,batch_mbboxes,batch_lbboxes

if __name__ == "__main__":
    
    train_file = '/home/luoyuhao/Tools/yolov3/tensorflow-yolov3/data/images/train.txt'
    yolo_data = Dataset(train_file)
    with tf.Session() as sess:
        for epoch in range(1):
            #sess.run(iterator.initializer)
            sess.run(yolo_data.train_data_init_op)
            for _ in range(3):
                img,s_label,m_label,l_label,s_box,m_box,l_box = sess.run(yolo_data.next_element)
                print(img.shape)
                print(s_label.shape)
                print(m_label.shape)
                print(l_label.shape)
                print(s_box.shape)
                print(m_box.shape)
                print(l_box.shape)





