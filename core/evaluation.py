from core.config import args
from pycocotools.cocoeval import COCOeval
from core.utils import rescale_bboxes
from tabulate import tabulate

import numpy as np
import logging
import progressbar
import pickle
import os
import json
import matplotlib.pyplot as plt


log = logging.getLogger()


#TODO refactor it to VOCEval and COCOEval with a common ancestor
class Evaluation(object):
    def __init__(self, net, loader, ckpt, conf_thresh=0.5, nms_thresh=0.3):
        self.net = net
        self.loader = loader
        self.gt = {}
        self.dets = {}
        self.ckpt = ckpt
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.show_img = False

        self.bridge_tp = 0
        self.airport_tp = 0
        self.car_tp = 0
        self.plane_tp = 0
        self.ship_tp = 0
        self.storangetank_tp = 0

        self.bridge_fn = 0
        self.airport_fn = 0
        self.car_fn = 0
        self.plane_fn = 0
        self.ship_fn = 0
        self.storangetank_fn = 0

        self.bridge_loss = 0
        self.airport_loss = 0
        self.car_loss = 0
        self.plane_loss = 0
        self.ship_loss = 0
        self.storangetank_loss = 0

        self.bridges = 0
        self.airports = 0
        self.cars = 0
        self.planes = 0
        self.ships = 0
        self.storangetanks = 0

        self.bridge_alert = 0
        self.airport_alert = 0
        self.car_alert = 0
        self.plane_alert = 0
        self.ship_alert = 0
        self.storangetank_alert = 0
        

    def evaluate_network(self, eval_first_n):
        filenames = self.loader.get_filenames()[:eval_first_n]
        self.gt = {cid: {} for cid in range(1, self.loader.num_classes)}
        self.dets = {cid: [] for cid in range(1, self.loader.num_classes)}

        start = 0
        cache = 'datasets/EvalCache/%s_%i.pickle' % (args.run_name, self.ckpt)
        if os.path.exists(cache) and not self.show_img:
            log.info("Found a partial eval cache: %s", cache)
            with open(cache, 'rb') as f:
                self.gt, self.dets, start = pickle.load(f)

        bar = progressbar.ProgressBar()
        for i in bar(range(start, len(filenames))):
            self.process_image(filenames[i], i)
            if i % 10 == 0 and i > 0 and not self.show_img:
                with open(cache, 'wb') as f:
                    pickle.dump((self.gt, self.dets, i), f, pickle.HIGHEST_PROTOCOL)
        table = []
        table.append(("bridge", self.bridge_tp, self.bridge_fn, self.bridge_loss, self.bridge_alert, self.bridges))
        table.append(("airport", self.airport_tp, self.airport_fn, self.airport_loss, self.airport_alert, self.airports))
        table.append(("car", self.car_tp, self.car_fn, self.car_loss, self.car_alert, self.cars))
        table.append(("plane", self.plane_tp, self.plane_fn, self.plane_loss, self.plane_alert, self.planes))
        table.append(("ship", self.ship_tp, self.ship_fn, self.ship_loss, self.ship_alert, self.ships))
        table.append(("storangetank", self.storangetank_tp, self.storangetank_fn, self.storangetank_loss, self.storangetank_alert, self.storangetanks))
        x = tabulate(table, headers=["category", "tp", "fn", "loss", "alert", "total"])
        log.info("\n"+x)
        if not self.show_img:
            log.debug("Cached eval results %s after the end", cache)
            with open(cache, 'wb') as f:
                pickle.dump((self.gt, self.dets, len(filenames)), f, pickle.HIGHEST_PROTOCOL)

    def bboxes_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        
        boxes1[..., 2] = boxes1[..., 2] + boxes1[..., 0]
        boxes1[..., 3] = boxes1[..., 3] + boxes1[..., 1]
        boxes2[..., 2] = boxes2[..., 2] + boxes2[..., 0]
        boxes2[..., 3] = boxes2[..., 3] + boxes2[..., 1]

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = 1.0 * inter_area / union_area

        return ious

    def process_image(self, name, img_id):
        img, scale = self.loader.load_image(name)
        gt_bboxes, gt_cats, _, _ = self.loader.read_annotations(name)
        proposals = self.loader.read_proposals(name)
        proposals = rescale_bboxes(proposals, scale)
        gt_bboxes = rescale_bboxes(gt_bboxes, scale)

        # for cid in np.unique(gt_cats):
        #     mask = (gt_cats == cid)
        #     bbox = gt_bboxes[mask]
        #     diff = difficulty[mask]
        #     det = np.zeros(len(diff), dtype=np.bool)
        #     self.gt[cid][img_id] = {'bbox': bbox, 'difficult': diff, 'det': det}

        det_cats, det_probs, det_bboxes = self.net.detect(img, proposals,
                                                          conf_thresh=self.conf_thresh,
                                                          nms_thresh=self.nms_thresh)
        _det_bboxes = det_bboxes.copy()
        _det_cats = det_cats.copy()
        _gt_bboxes = gt_bboxes.copy()
        _gt_cats = gt_cats.copy()
        for i in range(len(gt_bboxes)):
            if gt_cats[i] == 1:
                self.bridges += 1
            if gt_cats[i] == 2:
                self.airports += 1
            if gt_cats[i] == 3:
                self.cars += 1
            if gt_cats[i] == 4:
                self.planes += 1
            if gt_cats[i] == 5:
                self.ships += 1
            if gt_cats[i] == 6:
                self.storangetanks += 1

            if len(det_bboxes)==0:
                if gt_cats[i] == 1:
                    self.bridge_loss += 1
                if gt_cats[i] == 2:
                    self.airport_loss += 1
                if gt_cats[i] == 3:
                    self.car_loss += 1
                if gt_cats[i] == 4:
                    self.plane_loss += 1
                if gt_cats[i] == 5:
                    self.ship_loss += 1
                if gt_cats[i] == 6:
                    self.storangetank_loss += 1
                continue

            else:
                ious = self.bboxes_iou(gt_bboxes[i], det_bboxes)
                index = np.argmax(ious)
                iou = ious[index]
                if iou>0.1:
                    if det_cats[index]==gt_cats[i]:
                        if gt_cats[i] == 1:
                            self.bridge_tp += 1
                        if gt_cats[i] == 2:
                            self.airport_tp += 1
                        if gt_cats[i] == 3:
                            self.car_tp += 1
                        if gt_cats[i] == 4:
                            self.plane_tp += 1
                        if gt_cats[i] == 5:
                            self.ship_tp += 1
                        if gt_cats[i] == 6:
                            self.storangetank_tp += 1
                        del det_cats[index]
                        del det_bboxes[index]
                    else:
                        if det_cats[index] == 1:
                            self.bridge_fn += 1
                        if det_cats[index] == 2:
                            self.airport_fn += 1
                        if det_cats[index] == 3:
                            self.car_fn += 1
                        if det_cats[index] == 4:
                            self.plane_fn += 1
                        if det_cats[index] == 5:
                            self.ship_fn += 1
                        if det_cats[index] == 6:
                            self.storangetank_fn += 1
                        del det_cats[index]
                        del det_bboxes[index]
                else:
                    if gt_cats[i] == 1:
                        self.bridge_loss += 1
                    if gt_cats[i] == 2:
                        self.airport_loss += 1
                    if gt_cats[i] == 3:
                        self.car_loss += 1
                    if gt_cats[i] == 4:
                        self.plane_loss += 1
                    if gt_cats[i] == 5:
                        self.ship_loss += 1
                    if gt_cats[i] == 6:
                        self.storangetank_loss += 1
        for i in range(len(det_bboxes)):
            if det_cats[i] == 1:
                self.bridge_alert += 1
            if det_cats[i] == 2:
                self.airport_alert += 1
            if det_cats[i] == 3:
                self.car_alert += 1
            if det_cats[i] == 4:
                self.plane_alert += 1
            if det_cats[i] == 5:
                self.ship_alert += 1
            if det_cats[i] == 6:
                self.storangetank_alert += 1
        if self.show_img:
            # _det_bboxes.extend(_gt_bboxes)
            # _det_cats.extend(_gt_cats)
            visualize(img, _det_bboxes, _det_cats, self.loader)

        # for i in range(len(det_cats)):
        #     self.dets[det_cats[i]].append((img_id, det_probs[i]) + tuple(det_bboxes[i]))
        # print(self.dets[2])

    def eval_category(self, cid):
        cgt = self.gt[cid]
        cdets = np.array(self.dets[cid])
        if (cdets.shape == (0, )):
            return None, None
        scores = cdets[:, 1]
        sorted_inds = np.argsort(-scores)
        image_ids = cdets[sorted_inds, 0].astype(int)
        BB = cdets[sorted_inds]

        npos = 1
        for img_gt in cgt.values():
            img_gt['det'] = np.zeros(len(img_gt['difficult']), dtype=np.bool)
            npos += np.sum(~img_gt['difficult'])

        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            ovmax = -np.inf
            if image_ids[d] in cgt:
                R = cgt[image_ids[d]]
                bb = BB[d, 2:].astype(float)

                BBGT = R['bbox'].astype(float)

                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 0] + BBGT[:, 2], bb[0] + bb[2])
                iymax = np.minimum(BBGT[:, 1] + BBGT[:, 3], bb[1] + bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih

                # union
                uni = (bb[2] * bb[3] + BBGT[:, 2] * BBGT[:, 3] - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > 0.5:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = True
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float32).eps)
        return rec, prec


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):

            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


class COCOEval(Evaluation):
    def __init__(self, net, loader, ckpt, conf_thresh=0.5, nms_thresh=0.3):
        super().__init__(net, loader, ckpt, conf_thresh, nms_thresh)
        self.filename = '/home/lear/kshmelko/scratch/coco_eval_{}.json'.format(args.run_name)

    def process_image(self, img_id):
        img, scale = self.loader.load_image(img_id)
        gt_bboxes, gt_cats = self.loader.read_annotations(img_id)[:2]
        proposals = self.loader.read_proposals(img_id)
        proposals = rescale_bboxes(proposals, scale)
        gt_bboxes = rescale_bboxes(gt_bboxes, scale)

        det_cats, det_probs, det_bboxes = self.net.detect(img, proposals,
                                                          conf_thresh=self.conf_thresh,
                                                          nms_thresh=self.nms_thresh)

        detections = []
        for j in range(len(det_cats)):
            obj = {}
            obj['bbox'] = list(map(float, det_bboxes[j]/scale))
            obj['score'] = float(det_probs[j])
            obj['image_id'] = img_id
            obj['category_id'] = self.loader.ids_to_coco_ids[det_cats[j]]
            detections.append(obj)
        return detections

    def compute_ap(self):
        coco_res = self.loader.coco.loadRes(self.filename)

        cocoEval = COCOeval(self.loader.coco, coco_res)
        cocoEval.params.imgIds = self.image_ids
        cocoEval.params.catIds = self.loader.included_coco_ids
        cocoEval.params.useSegm = False

        ev_res = cocoEval.evaluate()
        acc = cocoEval.accumulate()
        summarize = cocoEval.summarize()

    def evaluate_network(self, eval_first_n):
        detections = []

        start = 0
        cache = '%sEvalCache/%s_%i.pickle' % (self.loader.root, args.run_name,
                                              self.ckpt)
        if os.path.exists(cache):
            log.info("Found a partial eval cache: %s", cache)
            with open(cache, 'rb') as f:
                detections, start = pickle.load(f)

        bar = progressbar.ProgressBar()
        self.image_ids = list(sorted(self.loader.coco.getImgIds()))[:eval_first_n]
        for i in bar(range(start, len(self.image_ids))):
            img_id = self.image_ids[i]
            detections.extend(self.process_image(img_id))
            if i % 10 == 0 and i > 0:
                with open(cache, 'wb') as f:
                    pickle.dump((detections, i), f, pickle.HIGHEST_PROTOCOL)

        with open(self.filename, 'w') as f:
            json.dump(detections, f)
        self.compute_ap()


def visualize(image, bboxes, cat_ids, loader, color='blue', scores=None):
    fig = plt.figure(0)
    plt.cla()
    plt.clf()
    plt.imshow(image)
    ax = plt.gca()
    for i in range(len(cat_ids)):
        bbox = bboxes[i]
        cat = loader.ids_to_cats[cat_ids[i]]
        if scores is None:
            title = cat
        else:
            title = '{:s} {:.3f}'.format(cat, scores[i])
        ax.add_patch(plt.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            fill=False,
            edgecolor='red',
            linewidth=2))
        ax.text(bbox[0],
                bbox[1] - 2,
                title,
                bbox=dict(facecolor=color, alpha=0.5),
                fontsize=14,
                color='white')
    plt.show()
