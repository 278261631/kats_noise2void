import numpy as np
from n2v.models import N2V
from astropy.io import fits
import glob
import os
import matplotlib.pyplot as plt
import tensorflow as tf

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
    fits_files = glob.glob(os.path.join(data_path, "*.fit*"))
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

# 加载测试数据
data_path = "real_data"
X_test = load_fits_images(data_path)

# 检查数据是否加载成功
if len(X_test) == 0:
    raise ValueError("没有成功加载任何测试图像数据。请检查数据路径和文件格式。")

print(f"成功加载的测试数据形状: {X_test.shape}")

# 数据预处理
X_test = X_test[..., np.newaxis]  # 添加通道维度
print(f"预处理后的测试数据形状: {X_test.shape}")

# 加载训练好的模型
model = N2V(config=None, name='n2v_astronomical_weights_best', basedir='models')
print("模型加载成功!")

# 对测试数据进行预测
print("开始对测试数据进行预测...")
# 为了节省时间和内存，我们可以只选择一部分数据进行测试
num_test_images = min(3, len(X_test))  # 测试前3张图像
X_test_subset = X_test[:num_test_images]

# 进行预测
X_pred = []
for i in range(num_test_images):
    print(f"预测第{i+1}/{num_test_images}张图像...")
    # 对图像进行分块预测
    # 由于GPU内存限制，使用分块预测
    # 设置较小的块大小和步长
    patch_size = 128
    stride = 96  # 重叠部分用于平滑拼接
    
    # 获取图像尺寸
    h, w = X_test_subset[i].shape[:2]
    
    # 计算块的数量
    num_h = (h - patch_size) // stride + 1
    num_w = (w - patch_size) // stride + 1
    
    # 初始化预测结果
    pred = np.zeros_like(X_test_subset[i])
    weights = np.zeros_like(X_test_subset[i])
    
    # 分块预测
    for i_h in range(num_h):
        for i_w in range(num_w):
            # 计算块的起始和结束位置
            start_h = i_h * stride
            end_h = start_h + patch_size
            start_w = i_w * stride
            end_w = start_w + patch_size
            
            # 提取块
            patch = X_test_subset[i][start_h:end_h, start_w:end_w, :]
            
            # 预测块
            patch_pred = model.predict(patch, axes='YXC')
            
            # 累加预测结果
            pred[start_h:end_h, start_w:end_w, :] += patch_pred
            weights[start_h:end_h, start_w:end_w, :] += 1
    
    # 归一化重叠区域
    pred = pred / np.maximum(weights, 1e-8)
    X_pred.append(pred)

X_pred = np.array(X_pred)
print(f"预测完成，预测结果形状: {X_pred.shape}")

# 保存结果为FITS格式和可视化
print("开始保存结果...")
# 创建保存结果的目录
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

for i in range(num_test_images):
    # 保存为FITS格式
    fits_save_path = os.path.join(results_dir, f"result_{i+1}.fit")
    # 移除通道维度 (从 (h, w, 1) 变为 (h, w))
    pred_data = X_pred[i][..., 0].astype(np.float32)
    # 创建FITS文件
    hdu = fits.PrimaryHDU(pred_data)
    hdul = fits.HDUList([hdu])
    hdul.writeto(fits_save_path, overwrite=True)
    print(f"结果已保存为FITS格式至: {fits_save_path}")

    # 可视化结果
    print("开始可视化结果...")
    # 随机选择一个区域进行可视化
    h, w = X_test_subset[i].shape[:2]
    start_h = np.random.randint(0, h - 200)
    start_w = np.random.randint(0, w - 200)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(X_test_subset[i, start_h:start_h+200, start_w:start_w+200, 0], cmap='gray')
    axes[0].set_title('原始图像')
    axes[1].imshow(X_pred[i, start_h:start_h+200, start_w:start_w+200, 0], cmap='gray')
    axes[1].set_title('去噪后图像')
    
    # 保存PNG图像
    plt.tight_layout()
    png_save_path = os.path.join(results_dir, f"result_{i+1}.png")
    plt.savefig(png_save_path)
    print(f"结果图像已保存为PNG格式至: {png_save_path}")
    
    # 显示图像
    plt.show()

print("测试完成!")