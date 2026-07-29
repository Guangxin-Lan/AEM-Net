import math
import h5py
import os
import glob
import random
from obspy import read
from scipy.signal import resample
import numpy as np


RECEIVERS = np.array([
    [83, -155, 0], [-29, -63, 0], [-162, 81, 0], [4, -170, 0], [-167, -4, 0], [-75, -167, 0],
    [-173, -87, 0], [-132, -159, 0],

    [233, -212, 48], [170, -148, 48], [106, -85, 48], [42, -21, 48], [-21, 42, 48], [-85, 106, 48],
    [-148, 170, 48], [-212, 233, 48], [-212, 233, 0], [-163, 184, 0], [-113, 134, 0], [-64, 85, 0],
    [-14, 35, 0], [35, -14, 0], [85, -64, 0], [134, -113, 0], [184, -162, 0], [233, -212, 0],

    [144, 140, 0], [83, -165, 48], [-45, -168, 48], [-91, -86, 48], [-110, 22, 48], [-180, -18, 48]
], dtype=int)


def random_source(low=[-250, -250, 0], high=[250, 250, 48],
                  forbidden_coords=None, max_retry=10000):
    """
    随机生成声源坐标，避开传感器坐标。
    low, high 为 xyz 的整数范围，包含边界。
    """
    if forbidden_coords is None:
        forbidden_set = set()
    else:
        forbidden_coords = np.asarray(forbidden_coords, dtype=int)
        forbidden_set = {tuple(coord) for coord in forbidden_coords}

    for _ in range(max_retry):
        xyz = np.array([
            np.random.choice(np.arange(low[0], high[0] + 1, 1)),
            np.random.choice(np.arange(low[1], high[1] + 1, 1)),
            np.random.randint(low[2], high[2] + 1)
        ], dtype=int)

        if tuple(xyz) not in forbidden_set:
            return xyz

    raise RuntimeError("random_source: 达到最大重试次数，可能可用位置过少")


def get_random_xyz(velocity, receivers):
    """
    velocity: 波速，单位 mm/s，可以是数值或 'random'
    receivers: 传感器坐标，形状为 (N, 3)
    """
    if velocity == "random":
        velocity = np.random.choice(np.arange(4000, 6000, 100)) * 1000

    receivers = np.asarray(receivers, dtype=float)

    source = random_source(
        low=[-250, -250, 0],
        high=[250, 250, 48],
        forbidden_coords=receivers
    )

    results = []

    for i, receiver in enumerate(receivers, 1):
        xy_distance = math.sqrt(
            (source[0] - receiver[0]) ** 2 +
            (source[1] - receiver[1]) ** 2
        )

        distance = np.linalg.norm(receiver - source)
        travel_time = distance / velocity

        results.append([
            i,
            source.tolist(),
            receiver.tolist(),
            distance,
            travel_time,
            xy_distance
        ])

    return results


def resampling_rate_sac(sac_file, new_sampling_rate, overwrite=True):
    """
    SAC 文件重采样。
    overwrite=True 覆盖原文件；
    overwrite=False 存为 rs_原文件名。
    """
    st = read(sac_file)
    original_sampling_rate = st[0].stats.sampling_rate

    new_npts = int(len(st[0].data) * new_sampling_rate / original_sampling_rate)
    resampled_data = resample(st[0].data, new_npts)

    st[0].data = resampled_data
    st[0].stats.sampling_rate = new_sampling_rate

    if overwrite:
        output_file = sac_file
    else:
        path, filename = os.path.split(sac_file)
        output_file = os.path.join(path, f"rs_{filename}")

    st.write(output_file, format="SAC")
    print(f"文件已重采样并保存为: {output_file}")


def crop_or_pad(data, target_length):
    """
    数据统一长度：
    长了截断，短了补 0。
    """
    data = np.asarray(data, dtype=np.float32)

    if len(data) > target_length:
        data = data[:target_length]
    elif len(data) < target_length:
        pad_length = target_length - len(data)
        data = np.pad(data, (0, pad_length), mode='constant')

    return data


def random_crop_or_pad(data, target_length):
    """
    噪声数据统一长度：
    长了随机截取，短了补 0。
    """
    data = np.asarray(data, dtype=np.float32)

    if len(data) > target_length:
        start = random.randint(0, len(data) - target_length)
        data = data[start:start + target_length]
    elif len(data) < target_length:
        pad_length = target_length - len(data)
        data = np.pad(data, (0, pad_length), mode='constant')

    return data


def normalize(data):
    """
    按最大绝对值归一化。
    """
    data = np.asarray(data, dtype=np.float32)
    max_data = np.max(np.abs(data))
    data = data / (max_data + 1e-12)
    return data


def load_one_noise(noise_folder_list, target_length):
    """
    随机读取一个真实噪声 SAC 文件，并进行：
    1. 随机截取或补零；
    2. 最大值归一化；
    3. 随机反相。
    """
    noise_file = random.choice(noise_folder_list)

    if not os.path.exists(noise_file):
        raise FileNotFoundError(f"找不到噪声文件: {noise_file}")

    tr = read(noise_file)[0]
    noise = tr.data.astype(np.float32, copy=False)

    noise = random_crop_or_pad(noise, target_length)
    noise = normalize(noise)

    if random.random() < 0.5:
        noise = -noise

    return noise


def get_1sampdata(wave_folder_path, noise_folder_list,
                  velocity, receivers, num_event, samp_rate):
    """
    生成一个样本：
    85% 为事件样本；
    15% 为纯噪声样本。
    """
    pre_target_length = 1000
    target_length = 1024

    # =========================
    # 85%：事件样本
    # =========================
    if random.random() < 0.85:
        results = get_random_xyz(velocity, receivers)

        all_data = []
        all_src_xyz = []
        all_stn_xyz = []
        all_distance = []
        all_travel_time = []
        all_xy_distance = []
        all_p_arr_npts = []

        sampling_rate = []
        delta = []

        # 计算所有台站理论走时
        t_time = []
        for i, value in enumerate(results):
            travel_time = results[i][4]
            t_time.append(travel_time)

        min_travel_time = min(t_time)
        min_index = t_time.index(min_travel_time)

        # 让最早到达台站的 P 波随机落在 50 到 950 点之间
        random_num = random.randint(50, 950)
        time_random = random_num * (1 / samp_rate)
        pre_time = time_random - results[min_index][4]

        # 同一个事件样本共用一个噪声强度
        noise_scale = np.random.uniform(0, 0.2)

        for i, value in enumerate(results):
            src_xyz = results[i][1]
            stn_xyz = results[i][2]
            distance = results[i][3]
            travel_time = results[i][4]
            xy_distance = results[i][5]

            random_number = random.randint(0, num_event)
            random_sac_file = os.path.join(
                wave_folder_path,
                f'idx{random_number}_{i}.sac'
            )

            if not os.path.exists(random_sac_file):
                raise FileNotFoundError(f"找不到事件波形文件: {random_sac_file}")

            s1 = read(random_sac_file)

            p_arr_time = s1[0].stats.sac.t0
            starttime = s1[0].stats.starttime

            st = s1.slice(
                starttime=starttime + p_arr_time - travel_time - pre_time,
                endtime=starttime + p_arr_time - travel_time - pre_time +
                        pre_target_length * s1[0].stats.delta
            )

            data = st[0].data
            data = crop_or_pad(data, pre_target_length)
            data = normalize(data)

            if random.random() < 0.5:
                data = -data

            all_data.append(data)
            all_src_xyz.append(src_xyz)
            all_stn_xyz.append(stn_xyz)

            sampling_rate.append(s1[0].stats.sampling_rate)
            delta.append(s1[0].stats.delta)

            all_distance.append(distance)
            all_xy_distance.append(xy_distance)
            all_p_arr_npts.append((travel_time + pre_time) * s1[0].stats.sampling_rate)
            all_travel_time.append(travel_time)

        # =========================
        # 距离衰减幅值
        # =========================
        relative_mag = 0

        expo = (
            np.asarray(relative_mag, dtype=float)
            - 1.7427 * np.log10(all_distance)
            + 8.5633
        )

        A = np.power(10.0, expo)
        A_max_val = A.max()
        A_norm = A / (A_max_val + 1e-12)

        all_data = np.asarray(all_data, dtype=np.float32)
        A_norm = np.asarray(A_norm, dtype=np.float32)

        scaled_data = all_data * A_norm.reshape(-1, 1)

        # =========================
        # 事件样本加两段真实噪声
        # =========================
        final_data = []

        for i in range(len(scaled_data)):
            data = np.asarray(scaled_data[i], dtype=np.float32)

            # 第一段真实噪声
            noise1 = load_one_noise(noise_folder_list, pre_target_length)

            # 第二段真实噪声
            noise2 = load_one_noise(noise_folder_list, pre_target_length)

            # 两段噪声各占一半强度
            data = data + noise1 * noise_scale / 2 + noise2 * noise_scale / 2

            # 统一到 1024 点
            data = crop_or_pad(data, target_length)

            final_data.append(data)

        final_data = np.asarray(final_data, dtype=np.float32)
        final_data = normalize(final_data)

        valid_p = [v for v in all_p_arr_npts if v >= 0]
        min_p_arr = float(min(valid_p)) if len(valid_p) > 0 else -1.0

        sample_dict = {
            "data": final_data,
            "srcxyz": all_src_xyz,
            "stnxyz": all_stn_xyz,
            "npts": target_length,
            "sample_rate": sampling_rate[0],
            "delta": delta[0],
            "distance": all_distance,
            "p_arr": all_p_arr_npts,
            "travel_time": all_travel_time,
            "xy_distance": all_xy_distance,
            "norm_A": A_norm,
            "min_p_arr": min_p_arr
        }

    # =========================
    # 15%：纯噪声样本
    # =========================
    else:
        final_data = []

        for i in range(32):
            # 第一段真实噪声
            data1 = load_one_noise(noise_folder_list, pre_target_length)

            # 第二段真实噪声
            data2 = load_one_noise(noise_folder_list, pre_target_length)

            # 纯噪声样本中第二段噪声乘随机系数
            noise_scale = np.random.uniform(0, 0.05)
            data = data1 + data2 * noise_scale

            data = crop_or_pad(data, target_length)
            data = normalize(data)

            final_data.append(data)

        final_data = np.asarray(final_data, dtype=np.float32)
        final_data = normalize(final_data)

        sample_dict = {
            "data": final_data,
            "srcxyz": -1,
            "stnxyz": -1,
            "npts": target_length,
            "sample_rate": samp_rate,
            "delta": 1 / samp_rate,
            "distance": -1,
            "p_arr": -1,
            "travel_time": -1,
            "xy_distance": -1,
            "norm_A": -1,
            "min_p_arr": -1
        }

    return sample_dict


def gen_train_h5(out_file, num_dataset, wave_folder_path, noise_folder_list,
                 velocity, receivers, num_event, samp_rate):
    """
    生成 H5 数据集。
    """

    out_dir = os.path.dirname(out_file)
    if out_dir != "" and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with h5py.File(out_file, 'w') as hf_o:
        for i in range(num_dataset):
            sample_list = get_1sampdata(
                wave_folder_path=wave_folder_path,
                noise_folder_list=noise_folder_list,
                velocity=velocity,
                receivers=receivers,
                num_event=num_event,
                samp_rate=samp_rate
            )

            tmp = hf_o.create_dataset('idx' + str(i), data=sample_list["data"])

            tmp.attrs['srcxyz'] = sample_list["srcxyz"]
            tmp.attrs['stnxyz'] = sample_list["stnxyz"]
            tmp.attrs['npts'] = sample_list["npts"]
            tmp.attrs['sample_rate'] = sample_list["sample_rate"]
            tmp.attrs['delta'] = sample_list["delta"]
            tmp.attrs['distance'] = sample_list["distance"]
            tmp.attrs['p_arr'] = sample_list["p_arr"]
            tmp.attrs['travel_time'] = sample_list["travel_time"]
            tmp.attrs['xy_distance'] = sample_list["xy_distance"]
            tmp.attrs['norm_A'] = sample_list["norm_A"]
            tmp.attrs['min_p_arr'] = sample_list["min_p_arr"]

            # 防止纯噪声样本 srcxyz=-1 时报错
            if isinstance(sample_list["srcxyz"], list):
                print('idx', i, 'event sample, srcxyz:', sample_list["srcxyz"][0])
            else:
                print('idx', i, 'noise sample, srcxyz:', sample_list["srcxyz"])


# =========================
# 参数设置
# =========================

velocity = 4500 * 1000  # mm/s
num_receivers = 32
num_event = 39
new_sampling_rate = 3000000

wave_folder_path = r"E:\lgx\code\AEMNet\locate\iog_data\iog_train_highSNR_basewave_decay"
noise_folder_path = r'E:\lgx\code\AEMNet\locate\iog_data\noise'

noise_folder_list = glob.glob(os.path.join(noise_folder_path, '*.sac'))

if len(noise_folder_list) == 0:
    raise RuntimeError(f"没有找到噪声 SAC 文件，请检查路径: {noise_folder_path}")

out_file = r'E:\lgx\code\AEMNet\data/train_dataset_xy1_100000_det_0.2.h5'


# =========================
# 生成 H5 数据集
# =========================

gen_train_h5(
    out_file=out_file,
    num_dataset=100000,
    wave_folder_path=wave_folder_path,
    noise_folder_list=noise_folder_list,
    velocity=velocity,
    receivers=RECEIVERS,
    num_event=num_event,
    samp_rate=new_sampling_rate
)