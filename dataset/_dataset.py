import abc
import imageio


class Dataset:
    def __init__(self, data_type, run_type, dataset, path_zflat, factor, bd_factor):
        self.data_type = data_type
        self.dataset = dataset
        self.run_type = run_type
        self.data_dir = f'./dataset/{data_type}/{dataset}'
        self.path_zflat = path_zflat
        self.factor = factor
        self.bd_factor = bd_factor

    @property
    def intrinsic_matrix(self):
        raise NotImplementedError

    @property
    def hwf(self):
        raise NotImplementedError

    @property
    def extrinsic_matrix(self):
        raise NotImplementedError

    @property
    def w2c(self):
        raise NotImplementedError

    @property
    def c2w(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_imgs(self):
        """
        Load Image files and return them in numpy.ndarray
            - Shape : (H, W, 3)
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_matrices(self):
        raise NotImplementedError

    def get_intrinsic_matrix(self):
        raise NotImplementedError

    def update_factor_img(self):
        raise NotImplementedError

    def update_factor_boundary(self):
        raise NotImplementedError


def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)