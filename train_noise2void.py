import numpy as np
from n2v.models import N2VConfig, N2V
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from astropy.io import fits
import glob
import os
import tensorflow as tf
import os

# 检查CUDA是否可用
has_cuda = tf.test.is_built_with_cuda()
print(f"TensorFlow是否构建了CUDA支持: {has_cuda}")

# 强制使用GPU (如果可用)
if has_cuda:
    # 设置环境变量以优先使用GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 尝试配置GPU内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"成功找到并配置了{len(gpus)}个GPU")
            print(f"GPU设备名称: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")
    else:
        print("未找到GPU设备，但TensorFlow支持CUDA。请检查NVIDIA驱动和CUDA安装。")
else:
    print("TensorFlow未构建CUDA支持。请安装支持CUDA的TensorFlow版本。")

# 创建会话时指定GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 限制GPU内存使用比例
# 尝试创建会话
try:
    sess = tf.compat.v1.Session(config=config)
    print("成功创建GPU会话")
except Exception as e:
    print(f"创建GPU会话时出错: {e}")

# 加载FITS天文图像
def load_fits_images(data_path):
    # 获取数据路径的绝对路径
    abs_data_path = os.path.abspath(data_path)
    print(f"正在从路径加载FITS图像: {abs_data_path}")
    
    # 查找所有FITS文件
    fits_files = glob.glob(os.path.join(data_path, "*.fit*"))  # 注意这里修改为.fit*
    print(f"找到的FITS文件数量: {len(fits_files)}")
    print(f"找到的FITS文件列表: {fits_files}")
    
    images = []
    if len(fits_files) == 0:
        print("警告: 没有找到任何FITS文件!")
        return np.array([])
    
    for file in fits_files:
        try:
            with fits.open(file) as hdul:
                data = hdul[0].data.astype(np.float32)
                images.append(data)
                print(f"成功加载文件: {file}, 形状: {data.shape}")
        except Exception as e:
            print(f"加载文件 {file} 时出错: {e}")
    
    if len(images) == 0:
        print("警告: 没有成功加载任何图像数据!")
        return np.array([])
    
    return np.array(images)

# 数据路径
data_path = "data"
X = load_fits_images(data_path)

# 检查数据是否加载成功
if len(X) == 0:
    raise ValueError("没有成功加载任何图像数据。请检查数据路径和文件格式。")

print(f"成功加载的数据形状: {X.shape}")

# 数据预处理
X = X[..., np.newaxis]  # 添加通道维度
print(f"预处理后的数据形状: {X.shape}")

# 创建数据生成器
datagen = N2V_DataGenerator()
patches = datagen.generate_patches_from_list([X], shape=(64, 64))

# 分割训练和验证数据
X_train = patches[:int(0.9*len(patches))]
X_val = patches[int(0.9*len(patches)):]

# 配置模型 - 根据RTX 3050调整参数
config = N2VConfig(X_train, 
                   unet_kern_size=3,
                   train_steps_per_epoch=200,  # 减少步数以适应GPU内存
                   train_epochs=100,
                   train_loss='mse',
                   batch_size=32,  # 减小批量大小以适应RTX 3050显存
                   train_batch_size=32,
                   n2v_perc_pix=0.198,
                   n2v_patch_shape=(64, 64),
                   n2v_manipulator='uniform_withCP',
                   n2v_neighborhood_radius=5,
                   train_checkpoint='weights_best.weights.h5',
                   train_learning_rate=0.0002)  # 调整学习率以适应新的批量大小

# 创建和训练模型
# 修改模型名称，使其符合Keras的weights.h5格式要求
model = N2V(config, 'n2v_astronomical_weights_best', basedir='models')
print("开始训练模型...")
history = model.train(X_train, X_val)
print("模型训练完成!")