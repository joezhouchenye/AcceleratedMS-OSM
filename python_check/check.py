import numpy as np
import matplotlib.pyplot as plt


def read_complex_bin(filename, start_index=0, length=None):
    """
    读取 uint32 打包复数数据

    Parameters
    ----------
    filename : str
        文件名

    start_index : int
        起始复数采样点索引

    length : int or None
        读取长度（复数点数）
        None 表示读到文件结束
    """

    # 每个复数占4字节(uint32)
    offset = start_index * 4

    with open(filename, 'rb') as f:

        # 跳转到指定位置
        f.seek(offset)

        # 读取数据
        if length is None:
            raw = np.fromfile(f, dtype=np.uint32)
        else:
            raw = np.fromfile(f, dtype=np.uint32, count=length)

    # 低16位：实部
    real_part = raw & np.uint32(0xFFFF)

    # 高16位：虚部
    imag_part = raw >> np.uint32(16)

    # 转 float
    real_part = real_part.astype(np.float64)
    imag_part = imag_part.astype(np.float64)

    # 非零减32768
    real_part[real_part != 0] -= 32768
    imag_part[imag_part != 0] -= 32768

    # 组成复数
    data_complex = real_part + 1j * imag_part

    return data_complex


# =====================================
# 用户输入
# =====================================

filename = input("Input filename: ")

start_index = int(input("Input start index: "))

length_str = input("Input length (empty means all): ")

if length_str.strip() == "":
    length = None
else:
    length = int(length_str)

# =====================================
# 读取数据
# =====================================

data = read_complex_bin(filename, start_index, length)

# =====================================
# 幅值
# =====================================

magnitude = np.abs(data)

# =====================================
# 绘图
# =====================================

plt.figure(figsize=(12, 5))

plt.plot(magnitude)

plt.xlabel("Sample Index")
plt.ylabel("Magnitude")
plt.title("Signal Magnitude")

plt.grid(True)

plt.show()