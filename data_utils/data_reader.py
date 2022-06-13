
class Dataset():
    def __init_(self):
        self.idx=0
        self.examples = []
    
    def get_next(self):
        if self.idx>=len(self.examples):
            return None
        dp = self.examples[self.idx]
        self.idx+=1

        return dp
