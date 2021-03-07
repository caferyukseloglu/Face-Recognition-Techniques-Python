import os

class ReadFiles:
    def __init__(self):
        self.files = {}

    def getfiles(self):
        for root, dirs, files in os.walk("./ImageBasic/"):
            for name in files:
                if filter(lambda image: (image[-4:] == '.jpg' or image[-4:] == '.png'), os.path.join(root, name)):
                    if root in self.files:
                        if not isinstance(self.files[root],list):
                            self.files[root] = [self.files[root]]
                        self.files[root].append(name)
                    else:
                        self.files.update({root:name})
                
    def get_images(self):
        self.getfiles()
        return self.files