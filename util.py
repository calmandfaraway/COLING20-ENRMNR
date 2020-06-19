import os
import time
import logging
import math
import numpy as np


logger = logging.getLogger(__name__)

class LossClock:
    def __init__(self, loss_names, interval, silence=False):
        self.losses = {name: 0. for name in loss_names}
        self.interval = interval
        self.silence = silence
        self.last_time = time.time()
        self.cnt = 0
    
    def update(self, named_loss):
        assert type(named_loss) is dict
        for name in named_loss:
            assert name in self.losses
            self.losses[name] += named_loss[name]
        self.cnt += 1
        if self.cnt % self.interval == 0:
            now = time.time()
            time_cost = int(now - self.last_time)
            for name in self.losses:
                self.losses[name] /= self.interval
            loss_repr = '; '.join(["%s: %.4f" % (name, self.losses[name]) for name in self.losses])
            info = "[Steps] => %d. [Losses] => %s. [Time cost] => %d min %d s." % (self.cnt, loss_repr, time_cost // 60, time_cost % 60)
            if not self.silence:
                logger.info(info)
            for name in self.losses:
                self.losses[name] = 0.
            self.last_time = now

def create_logger(log_path, log_name, debug=False):
    logFormatter = logging.Formatter("[%(levelname)s] [%(asctime)s] -- %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    if not debug:
        logger_file = os.path.join(log_path, log_name)
        fileHandler = logging.FileHandler(logger_file)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    return rootLogger

def mrr(preds, labels):
    print('********eval:**********')
    print(preds)  # [tensor([0.0143, 0.0120, 0.0097, 0.0114], device='cuda:0')]
    print(labels)  # [tensor([0, 0, 0, 1], device='cuda:0')]
    sum_query, sum_rank_score = 0, 0.
    for pred, label in zip(preds, labels):
        print('*******')
        print(pred)
        print(label)
        if not isinstance(pred, list):  # isinstance 判断两个类型是否相同
            pred = pred.cpu().numpy().tolist()
        if not isinstance(label, list):
            label = label.cpu().numpy().tolist()
        tuples = list(zip(pred, label))
        print('tuples:')
        print(tuples)
        # [(0.014320275746285915, 0), (0.012020082212984562, 0), (0.009679405018687248, 0), (0.011412363499403, 1)]
        tuples.sort(key=lambda x: x[0], reverse=True)
        print(tuples)
        # [(0.014320275746285915, 0), (0.012020082212984562, 0), (0.011412363499403, 1), (0.009679405018687248, 0)]
        for idx, (_, l) in enumerate(tuples):
            if l == 1:
                sum_rank_score += 1/(idx + 1)
        sum_query += 1
    return sum_rank_score / sum_query

# p.shape=n.shape=(B,)
def max_margin_loss(p, n, anchor=1.0):
    # p=tensor([0.0282, 0.0282, 0.0282]  正例的得分
    # n=tensor([0.0592, 0.0412, 0.1387]  负例的得分
    margin = anchor - p + n
    return (margin * (margin > 0).float()).sum()

def sort_and_couple(labels: np.array, scores: np.array) -> np.array:
    """Zip the `labels` with `scores` into a single list."""
    couple = list(zip(labels, scores))
    return np.array(sorted(couple, key=lambda x: x[1], reverse=True))

def discountedcumulativegain(y_t, y_p, k):
    if k <= 0:
        return 0.
    coupled_pair = sort_and_couple(y_t, y_p)
    result = 0.
    for i, (label, score) in enumerate(coupled_pair):
        if i >= k:
            break
        if label > 0.:
            result += (math.pow(2., label) - 1.) / math.log(2. + i)
    return result

def ndcg(preds, labels, k):
    sum_query, sum_rank_score = 0, 0.
    for pred, label in zip(preds, labels):
        # print(pred)
        # print(label)
        if not isinstance(pred, list):  # isinstance 判断两个类型是否相同
            pred = pred.cpu().numpy().tolist()
        if not isinstance(label, list):
            label = label.cpu().numpy().tolist()
        idcg_val = discountedcumulativegain(label, label, k)
        dcg_val = discountedcumulativegain(label, pred, k)
        sum_rank_score += dcg_val / idcg_val if idcg_val != 0 else 0
        sum_query += 1
    return sum_rank_score / sum_query

def p(preds, labels, k):
    sum_query, sum_rank_score = 0, 0.
    for pred, label in zip(preds, labels):
        if not isinstance(pred, list):  # isinstance 判断两个类型是否相同
            pred = pred.cpu().numpy().tolist()
        if not isinstance(label, list):
            label = label.cpu().numpy().tolist()
        coupled_pair = sort_and_couple(label, pred)
        precision = 0.0
        for idx, (l, score) in enumerate(coupled_pair):
            if idx >= k:
                break
            if l > 0.:
                precision += 1.
        sum_rank_score += precision / k
        sum_query += 1
    return sum_rank_score / sum_query

def mapp(preds, labels):
    sum_query, sum_rank_score = 0, 0.
    for pred, label in zip(preds, labels):
        if not isinstance(pred, list):  # isinstance 判断两个类型是否相同
            pred = pred.cpu().numpy().tolist()
        if not isinstance(label, list):
            label = label.cpu().numpy().tolist()
        result = 0.
        pos = 0
        coupled_pair = sort_and_couple(label, pred)
        for idx, (l, score) in enumerate(coupled_pair):
            if l > 0.:
                pos += 1.
                result += pos / (idx + 1.)
        if pos == 0:
            sum_rank_score += 0
        else:
            sum_rank_score += result / pos
        sum_query += 1
    return sum_rank_score / sum_query
