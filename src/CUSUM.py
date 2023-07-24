# change detector block, receives rewards from environment and sends detections to the learner

class CUSUM:
    def __init__(self, M, eps, h):
        self.M = M  # first M valid samples are used to compute the reference point
        self.eps = eps
        self.h = h
        self.t = 0
        self.reference = 0
        self.g_plus = 0
        self.g_minus = 0

    def update(self, sample): 
        # return 1 if changes are detected, otherwise 0
        self.t += 1
        if self.t <= self.M:
            self.reference = (self.reference * (self.t - 1) + sample) / self.t
            return 0
        else:
            s_plus = (sample - self .reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            self.g_plus = max(0, self.g_plus + s_plus)
            self.g_minus = max(0, self.g_minus + s_minus)
            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):
        # reset all important variables in case of a detection
        self.t = 0
        self.g_minus = 0
        self.g_plus = 0