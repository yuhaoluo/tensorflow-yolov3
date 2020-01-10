from tqdm import tqdm
import tensorflow as tf

# from core.dataset import  Dataset
# yolo_data = Dataset('type')

# pbar_yolo = tqdm(yolo_data)
# for batch_data in pbar_yolo:
#     continue
#     #print(batch_data[0].shape)


## "./data/images/train.txt"



from core.lyhdata import Dataset
yolo_train_data = Dataset('train',num_parallel=1)
yolo_val_data = Dataset('val',num_parallel=1)
train_data = yolo_train_data.get_dataset()
val_data = yolo_val_data.get_dataset()
iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
train_init_op = iterator.make_initializer(train_data)
val_init_op = iterator.make_initializer(val_data)

#pbar = tqdm(range(yolo_train_data.batches_per_epoch))
#
with tf.Session() as sess:
    for epoch in range(1):
        #sess.run(yolo_data.train_data_init_op)
        sess.run(train_init_op)
        #sess.run(val_init_op)
        for _ in tqdm(range(yolo_train_data.batches_per_epoch)):
            img,s_label,m_label,l_label,s_box,m_box,l_box = sess.run(iterator.get_next())
            # print('img.shape',img.shape)
            # print('s_label.shape',s_label.shape)
            # print('m_label.shape',m_label.shape)
            # print('l_label.shape',l_label.shape)
            # print('l_box.shape',l_box.shape)
            