import numpy as np
import scipy

# Starting Point:
# https://github.com/shenyangHuang/TGB/blob/main/modules/heuristics.py

class MovingAverageMessages:
    def __init__(self, num_class, k):
        self.dict = {}
        self.time = {}
        self.num_class = num_class
        self.k = k

    def process_edges(self, srcs, dsts, ts, msgs):
        if srcs.nelement() == 0:
            return
        srcs = srcs.cpu().numpy()
        dsts = dsts.cpu().numpy()
        ts = ts.cpu().numpy()
        msgs = msgs[:, 0].cpu().numpy()

        for src, dst, t, msg in zip(srcs, dsts, ts, msgs):
            if src not in self.dict:
                self.dict[src] = np.zeros((self.num_class, self.k))

            self.dict[src][dst][:-1] = self.dict[src][dst][1:]
            self.dict[src][dst][-1] = msg

    def reset_state(self):
        self.dict = {}

    def query_dict(self, node_id, use_softmax):
        r"""
        Parameters:
            node_id: the node to query
        Returns:
            returns the last seen label of the node if it exists, if not return zero vector
        """
        if node_id in self.dict:
            mean = np.mean(self.dict[node_id], axis=1)
            if use_softmax:
                return scipy.special.softmax(mean)
            else: return mean
        else:
            return np.zeros(self.num_class)


