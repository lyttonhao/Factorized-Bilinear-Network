import logging
import mxnet as mx
import math


class FMConvScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, step, factor=1, fm_scale=1.0):
        super(FMConvScheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0
        self.fm_scale = fm_scale

    def __call__(self, num_update):
        """
        Call to schedule current learning rate
        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        
        lr = self.base_lr
        if self.cur_step_ind <= len(self.step)-1:
            if self.cur_step_ind == 0:
                if num_update < self.step[0]:
                    lr = (self.fm_scale + (1.0-self.fm_scale) * num_update / self.step[0]) * self.base_lr
                else:
                    self.cur_step_ind += 1
            elif num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= self.factor
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)

        return lr
