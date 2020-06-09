class Database:
    def __init__(self, name, location, sfreq=None, channels=None):
        self.name = name
        self.location = location
        self.sfreq = sfreq
        self.channels = channels
    
    def get_sfreq(self):
        return self.sfreq; 