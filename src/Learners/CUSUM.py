class CUSUM:
    """
    Cumulative sum algorithm(CUSUM) to actively detect changes in non-stationary environments with abrupt changes.
    It is the change detector block, receives rewards from the environment and sends detections to the learner

     Attributes:
        t: Time
        reference: Reference point
        g_plus: Cumulative positive deviation from the reference point
        g_minus: Cumulative negative deviation from the reference point
    """

    def __init__(self, M, eps, h):
        """
        Initializes CUSUM

        :param int M: Number of valid samples that are used to compute the reference point
        :param float eps: Epsilon parameter for CUSUM
        :param float h: Parameter over which a detection is flagged
        """

        self.M = M
        self.eps = eps
        self.h = h
        self.t = 0
        self.reference = 0
        self.g_plus = 0
        self.g_minus = 0

    def update(self, sample):
        """
        Updates the state of the change detector block. Returns 1 if a change is detected, 0 otherwise

        :param float sample: Observed reward
        :return: 1 if a change is detected, 0 otherwise
        :rtype: int
        """

        self.t += 1
        if self.t <= self.M:
            # First M samples are used to compute the reference point
            self.reference = (self.reference * (self.t - 1) + sample) / self.t
            return 0
        else:
            # Update the internal state
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            self.g_plus = max(0, self.g_plus + s_plus)
            self.g_minus = max(0, self.g_minus + s_minus)
            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):
        """
        Restart the detection algorithm by resetting the variables in case of a detection
        """

        self.t = 0
        self.g_minus = 0
        self.g_plus = 0
