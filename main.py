import dataset
import yaml

if __name__ == '__main__':
    # config = yaml.safe_load(open('./config.yml'))['llff']
    # dset = dataset.SyntheticDataset(**config)

    config = yaml.safe_load(open('./config.yml'))['llff']
    dset = dataset.LLFFDataset(**config)


    print(config)
    print(dset.intrinsic_matrix)
    print(dset.hwf)
    print(dset.img_shape)
    print()

