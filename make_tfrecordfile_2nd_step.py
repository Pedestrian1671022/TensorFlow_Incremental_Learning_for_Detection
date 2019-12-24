import os
import numpy as np
import tensorflow as tf
from core.voc_loader import VOCLoader
slim = tf.contrib.slim


def normalize_bboxes(bboxes, w, h):
    """rescales bboxes to [0, 1]"""
    new_bboxes = np.array(bboxes, dtype=np.float32)
    new_bboxes[:, [0, 2]] /= w
    new_bboxes[:, [1, 3]] /= h
    return new_bboxes


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float64_feature(value):
    """Wrapper for inserting float64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_bboxes_to_features(bboxes, key):
    xmin = bboxes[:, 0].tolist()
    ymin = bboxes[:, 1].tolist()
    xmax = (bboxes[:, 2] + bboxes[:, 0]).tolist()
    ymax = (bboxes[:, 3] + bboxes[:, 1]).tolist()
    return {
        key+'/bbox/ymin': _float64_feature(ymin),
        key+'/bbox/xmin': _float64_feature(xmin),
        key+'/bbox/ymax': _float64_feature(ymax),
        key+'/bbox/xmax': _float64_feature(xmax),
    }


def _convert_to_example(filename, image_buffer, proposals, bboxes, cats,
                        height, width):
    labels = cats.tolist()
    image_format = 'JPEG'
    objects = _convert_bboxes_to_features(bboxes, 'image/object')
    proposals = _convert_bboxes_to_features(proposals, 'image/proposal')

    img_features = {
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/object/class/label': _int64_feature(labels),
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer)),
    }
    example = tf.train.Example(features=tf.train.Features(feature={
        **img_features, **proposals, **objects
    }))
    return example


if __name__ == '__main__':
    loader = VOCLoader("train")
    print("Contains %i files" % len(loader.get_filenames()))

    writer = tf.python_io.TFRecordWriter("datasets/data.tfrecord")
    for i, f in enumerate(loader.get_filenames()):
        path = 'datasets/Images/%s.jpg' % (f)
        with tf.gfile.FastGFile(path, 'rb') as ff:
            image_data = ff.read()
        gt_bb, gt_cats, w, h = loader.read_annotations(f)
        print(gt_bb, gt_cats, w, h)
        gt_bb = normalize_bboxes(gt_bb, w, h)
        proposals = loader.read_proposals(f)
        proposals = normalize_bboxes(proposals, w, h)
        example = _convert_to_example(path, image_data, proposals, gt_bb, gt_cats, h, w)
        if i % 100 == 0:
            print("%i files are processed" % i)
        writer.write(example.SerializeToString())

    writer.close()
    print("Done")
