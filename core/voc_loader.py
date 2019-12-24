import cv2
import numpy as np
from scipy.io import loadmat
import xml.etree.ElementTree as ET

VOC_CATS = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor']


class VOCLoader():

    def __init__(self, split):
        self.num_proposals = 2000
        self.split = split
        cats = VOC_CATS
        self.cats_to_ids = dict(map(reversed, enumerate(cats)))
        self.ids_to_cats = dict(enumerate(cats))
        self.num_classes = len(cats)
        self.categories = cats[1:]

    def convert_and_maybe_resize(self, im, resize):
        scale = 1.0
        if resize:
            h, w, _ = im.shape
            scale = min(1000/max(h, w), 600/min(h, w))
            im = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)/255.0
        return im, scale

    def load_image(self, name, resize=True):
        im = cv2.imread('datasets/Images/%s.jpg' % (name))
        out = self.convert_and_maybe_resize(im, resize)
        return out

    def get_filenames(self):
        with open('datasets/%s.txt' % (self.split), 'r') as f:
            return f.read().split('\n')[:-1]

    def read_proposals(self, name):
        mat = loadmat('datasets/EdgeBoxesProposals/%s.mat' % (name))
        bboxes = mat['bbs'][:, :4]
        return bboxes[:self.num_proposals]

    def read_annotations(self, name):
        bboxes = []
        cats = []

        tree = ET.parse('datasets/Annotations/%s.xml' % (name))
        root = tree.getroot()
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        for obj in root.findall('object'):
            cat = self.cats_to_ids[obj.find('name').text]
            cats.append(cat)
            bbox_tag = obj.find('bndbox')
            x = int(bbox_tag.find('xmin').text)
            y = int(bbox_tag.find('ymin').text)
            w = int(bbox_tag.find('xmax').text)-x
            h = int(bbox_tag.find('ymax').text)-y
            bboxes.append((x, y, w, h))

        gt_cats = np.array(cats)
        gt_bboxes = np.array(bboxes)

        return gt_bboxes, gt_cats, width, height


def create_permutation(last_class):
    cats = list(VOC_CATS)
    i = cats.index(last_class)
    j = cats.index('tvmonitor')
    cats[i], cats[j] = cats[j], cats[i]
    cats[:-1] = sorted(cats[:-1])
    return cats


def class_stats(ids, start_id, end_id):
    common = set()
    for i in range(start_id, end_id+1):
        common = common | ids[i]
    print("Classes from {} to {} are in {} images".format(start_id, end_id, len(common)))


if __name__ == '__main__':
    print("Statistics per class: ")
    ids = {i: set() for i in range(1, 21)}
    loader = VOCLoader("total")
    total = 0

    for name in loader.get_filenames():
        gt_cats = loader.read_annotations(name)[1]
        for cid in gt_cats:
            ids[cid].add(name)
    for i in ids.keys():
        print("%s: %i" % (VOC_CATS[i], len(ids[i])))
        total += len(ids[i])
    print("TOTAL: %i" % total)

    class_stats(ids, 1, 2)
    class_stats(ids, 3, 3)
    class_stats(ids, 4, 4)
    class_stats(ids, 5, 5)
