import pickle
import os

class savingData():
    def __init__(self):
        os.makedirs("models", exist_ok=True)
    def save(self, title, value):
        with open(f'models/{title}', 'wb') as f:
            pickle.dump(value, f)
            print(f'Saved {title}')
