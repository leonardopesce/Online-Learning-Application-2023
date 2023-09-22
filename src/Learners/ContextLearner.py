class ContextLearner:
    def __init__(self, context, learner) -> None:
        self.context = context
        self.learner = learner

    def get_learner(self):
        return self.learner
    
    def get_context(self):
        return self.context
    
    @property
    def t(self):
        return self.learner.t
    
    @t.setter
    def t(self, value):
        self.learner.t = value
