import numpy as np
from n2v.models import N2VConfig, N2V
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from astropy.io import fits
import glob
import os

# 加载FITS天文图像
def load_fits_images(data_path):
    fits_files = glob.glob(os.path.join(data_path, "*.fits"))
    images = []
    for file in fits_files:
        with fits.open(file) as hdul:
            data = hdul[0].data.astype(np.float32)
            images.append(data)
    return np.array(images)

# 数据路径
data_path = "data"
X = load_fits_images(data_path)

# 数据预处理
X = X[..., np.newaxis]  # 添加通道维度

# 创建数据生成器
datagen = N2V_DataGenerator()
patches = datagen.generate_patches_from_list([X], shape=(64, 64))

# 分割训练和验证数据
X_train = patches[:int(0.9*len(patches))]
X_val = patches[int(0.9*len(patches)):]

# 配置模型
config = N2VConfig(X_train, 
                   unet_kern_size=3,
                   train_steps_per_epoch=400,
                   train_epochs=100,
                   train_loss='mse',
                   batch_size=128,
                   train_batch_size=128,
                   n2v_perc_pix=0.198,
                   n2v_patch_shape=(64, 64),
                   n2v_manipulator='uniform_withCP',
                   n2v_neighborhood_radius=5)

# 创建和训练模型
model = N2V(config, 'n2v_astronomical', basedir='models')
history = model.train(X_train, X_val)