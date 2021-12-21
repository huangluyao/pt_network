
import multiprocessing
import random
from itertools import cycle
from .utils import BatchSampler, default_collate, put_data
import numpy as np

class SimpleDataLoader:

    def __init__(self, dataset, batch_size, shuffle=False, num_workers=0, drop_last=False, collate_fn=None):
        num_workers = max(1, num_workers)
        self.dataset = dataset
        self.drop_last = drop_last
        self.index_queue = []
        sampler = range(len(self.dataset)) if not shuffle else random.sample(range(len(dataset)), len(dataset))
        self.batch_index_sampler = BatchSampler(sampler, batch_size, drop_last=drop_last)
        self.num_workers = num_workers
        self.worker_queue_index = cycle(range(num_workers))
        self.image_queue = multiprocessing.Queue()
        self.num_dataset = len(dataset)
        self.batch_size = batch_size
        if collate_fn is None:
            collate_fn = default_collate

        for i in range(num_workers):
            index_queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=put_data, args=(dataset, index_queue, self.image_queue, collate_fn))
            p.daemon = True
            p.start()
            self.index_queue.append(index_queue)
            pass

    def __iter__(self):
        self.reset()
        last_batch = 1 if not self.drop_last and self.num_dataset % self.batch_size else 0
        iter_num = self.num_dataset // self.batch_size + last_batch
        for i in range(iter_num):
            data = self.image_queue.get()
            yield data
            self.try_put_index()

    def __len__(self):
        last_batch = 1 if not self.drop_last and self.num_dataset % self.batch_size else 0
        iter_num = self.num_dataset // self.batch_size + last_batch
        return iter_num


    def reset(self):
        self.batch_index_iter = iter(self.batch_index_sampler)
        for _ in range(self.num_workers):
            self.try_put_index()

    def try_put_index(self):
        try:
            batch_index = next(self.batch_index_iter)
            worker_index = next(self.worker_queue_index)
            self.index_queue[worker_index].put(batch_index)
        except StopIteration:
            return


def few_shot_dataset_collate(batch):
    images = []
    labels = []

    # todo debug
    # for images in batch:
    #     for image in images[0]:
    #         img = np.transpose(image, [1, 2, 0])*255
    #         img = img.astype(np.int8)
    #         cv2.imshow('tse', img)
    #         cv2.waitKey()

    for img, label in batch:
        images.append(img)
        labels.append(label)

    images1 = np.array(images)[:, 0, :, :, :]
    images2 = np.array(images)[:, 1, :, :, :]
    images3 = np.array(images)[:, 2, :, :, :]
    images = np.concatenate([images1, images2, images3], 0)

    labels1 = np.array(labels)[:, 0]
    labels2 = np.array(labels)[:, 1]
    labels3 = np.array(labels)[:, 2]
    labels = np.concatenate([labels1, labels2, labels3], 0)
    gt = dict(gt_labels=labels)
    return images, gt


def detection_dataset_collate(batch):
    batch_img, batch_info = zip(*batch)

    batch_labels = dict()
    keys = batch_info[0].keys()
    for info in batch_info:
        for key in keys:
            if key in batch_labels:
                batch_labels[key].append(info[key])
            else:
                batch_labels[key] = [info[key]]
    batch_img = np.transpose(np.array(batch_img),[0, 3, 1, 2])
    return batch_img, batch_labels
