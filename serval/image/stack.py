import skimage


class ImageStack(object):
    @staticmethod
    def load(file_name, fov=0, z=0):
        return ImageStack(skimage.io.imread(file_name), fov=fov, z=z)

    def __init__(self, imgs, fov=0, z=0):
        self.imgs = imgs

        self.fov = fov

        self.z = z

    @property
    def num_frames(self):
        return self.imgs.shape[0]
