import abc


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
    
    @property
    def render_pose(self):
        raise NotImplementedError
    
    @property
    def test_i(self):
        raise NotImplementedError
    
    @property
    def train_i(self):
        raise NotImplementedError
    
    @property
    def val_i(self):
        raise NotImplementedError
    
    @property
    def near(self):
        raise NotImplementedError

    @property
    def far(self):
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
        """
        Return pose matrix(poses) and boundary values(bds)
        
        Sometimes, raw data is made of inverse of pose matrix(so to speak, c2w).
        In this case, Dataset will use "poses_inv" as a name of attr.
        
        Also, raw data can be stored in [-y, x, z] order(It comes from camera!)

            cf) Pose matrix = Intrinsic Matrix x Extrinsic Matrix
                            = w2c

        """
        raise NotImplementedError

    def get_intrinsic_matrix(self):
        raise NotImplementedError

    def update_factor_img(self):
        raise NotImplementedError

    def update_factor_boundary(self):
        raise NotImplementedError
