import math
import h5py
import os
import glob
import random
from matplotlib import pyplot as plt
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


# def random_source(low=[-250, -250, 0], high=[250, 250, 48], size=3, boundary_prob=0.05):
#     # 生成基础随机数（半开区间 [low, high)）
#     xyz = np.random.uniform(low=low, high=high, size=size)
#     # 对每个维度独立判断是否替换为边界值
#     for i in range(3):
#         if np.random.random() < boundary_prob:
#             xyz[i] = np.random.choice([low[i], high[i]])
#
#     return xyz


def random_source(low=[-250, -250, 0], high=[250, 250, 48],
                  forbidden_coords=None, max_retry=10000):
    """
    随机生成声源坐标，避开 forbidden_coords 中的传感器坐标。
    low, high 为 xyz 的整数范围（包含边界）。
    forbidden_coords: 形如 [[x1,y1,z1], [x2,y2,z2], ...] 的列表或 ndarray
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


def random_receivers(num_receivers=6, xy_range=(-250, 250), z_options=[0, 48], z_prob=[0.5, 0.5]):
    receivers = []
    min_xy, max_xy = xy_range

    for _ in range(num_receivers):
        while True:
            x = random.randint(min_xy, max_xy)
            y = random.randint(min_xy, max_xy)
            # 检查是否同时为边界值（在棱边上）
            if x not in (min_xy, max_xy) and y not in (min_xy, max_xy):
                break
        z = random.choices(z_options, weights=z_prob)[0]
        receivers.append([x, y, z])
    return receivers


def get_random_xyz(velocity, receivers):
    """
    velocity: 波速（可以是数值或 'random'）
    receivers: 外部传入的传感器坐标，形状 (N,3)
    """
    # 随机速度
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
        xy_distance = math.sqrt((source[0] - receiver[0]) ** 2 + (source[1] - receiver[1]) ** 2)
        distance = np.linalg.norm(receiver - source)
        time = distance / velocity  # s
        results.append([i, source.tolist(), receiver.tolist(),
                        distance, time, xy_distance])
    return results


def resampling_rate_sac(sac_file, new_sampling_rate, overwrite=True):
    # overwrite=True 覆盖原文件，overwrite=False 存为rs_原文件
    st = read(sac_file)
    original_sampling_rate = st[0].stats.sampling_rate

    new_npts = int(len(st[0].data) * new_sampling_rate / original_sampling_rate)
    resampled_data = resample(st[0].data, new_npts)

    st[0].data = resampled_data
    st[0].stats.sampling_rate = new_sampling_rate

    if overwrite:
        output_file = sac_file
    else:
        import os
        path, filename = os.path.split(sac_file)
        output_file = os.path.join(path, f"rs_{filename}")

    st.write(output_file, format="SAC")
    print(f"文件已重采样并保存为: {output_file}")


def get_1sampdata(wave_folder_path, noise_path, velocity, receivers, num_event, samp_rate):
    results = get_random_xyz(velocity, receivers)
    all_data = []
    all_src_xyz = []
    all_stn_xyz = []
    all_distance = []
    all_travel_time = []
    all_xy_distance = []
    all_p_arr_npts = []
    npts = []
    sampling_rate = []
    delta = []
    # random_time = np.random.randint(0, 400) * (1 / samp_rate)
    pre_target_length = 1000
    target_length = 1024

    t_time = []
    for i, value in enumerate(results):
        travel_time = results[i][4]
        t_time.append(travel_time)
    min_travel_time = min(t_time)
    min_index = t_time.index(min_travel_time)

    # time_500 = 500 * (1 / samp_rate)
    # pre_time = time_500 - results[min_index][4]
    random_num = random.randint(50, 950)
    time_random = random_num * (1 / samp_rate)
    pre_time = time_random - results[min_index][4]

    scale = np.random.uniform(0, 0.1)

    for i, value in enumerate(results):
        src_xyz = results[i][1]
        stn_xyz = results[i][2]
        distance = results[i][3]
        travel_time = results[i][4]
        xy_distance = results[i][5]

        folder_path = wave_folder_path
        random_number = random.randint(0, num_event)
        sac_files = os.path.join(folder_path, f'idx{random_number}_{i}.sac')
        random_sac_file = sac_files
        # print(random_sac_file)

        s1 = read(random_sac_file)
        p_arr_time = s1[0].stats.sac.t0
        starttime = s1[0].stats.starttime
        st = s1.slice(starttime=starttime + p_arr_time - travel_time - pre_time,
                      endtime=starttime + p_arr_time - travel_time - pre_time + pre_target_length * s1[0].stats['delta'])

        data = st[0].data

        if len(data) > pre_target_length:
            data = data[:pre_target_length]
        elif len(data) < pre_target_length:
            pad_length = pre_target_length - len(data)
            data = np.pad(data, (0, pad_length), mode='constant')

        max_data = np.max(np.abs(data))
        data = data / (max_data + 0.01e-100)

        if random.random() < 0.5:
            data = -data

        all_data.append(data)
        all_src_xyz.append(src_xyz)
        all_stn_xyz.append(stn_xyz)
        npts.append(s1[0].stats.npts)
        sampling_rate.append(s1[0].stats.sampling_rate)
        delta.append(s1[0].stats.delta)
        all_distance.append(distance)
        all_xy_distance.append(xy_distance)
        all_p_arr_npts.append((travel_time + pre_time) * s1[0].stats.sampling_rate)
        all_travel_time.append(travel_time)

    relative_mag = 0
    expo = np.asarray(relative_mag, dtype=float) - 1.7427 * np.log10(all_distance) + 8.5633
    A = np.power(10.0, expo)
    A_max_val = A.max()
    A_norm = (A / A_max_val + 0.01e-100)
    scaled_data = all_data * np.array(A_norm).reshape(-1, 1)

    final_data = []

    for i in range(len(scaled_data)):

        data = np.array(scaled_data[i])

        random_int1 = random.randint(0, 95)
        st1 = read(rf'{noise_path}/noise_{random_int1}.sac')
        noise1 = st1[0].data
        max_noise1 = max(abs(noise1))
        noise1 = noise1 / (max_noise1 + 0.01e-100)
        if random.random() < 0.5:
            noise1 = -noise1
        noise1 = noise1 * scale / 2

        random_int2 = random.randint(0, 95)
        st2 = read(rf'{noise_path}/noise_{random_int2}.sac')
        noise2 = st2[0].data
        max_noise2 = max(abs(noise2))
        noise2 = noise2 / (max_noise2 + 0.01e-100)
        if random.random() < 0.5:
            noise2 = -noise2
        noise2 = noise2 * scale / 2

        data = np.array(data) + noise1 + noise2

        if len(data) > target_length:
            data = data[:target_length]
        elif len(data) < target_length:
            pad_length = target_length - len(data)
            data = np.pad(data, (0, pad_length), mode='constant')

        final_data.append(data)

    final_data = np.array(final_data)

    max_final_data = np.max(np.abs(final_data))
    final_data = final_data / (max_final_data + 0.01e-100)

    sample_dict = {
        "data": final_data,
        "srcxyz": all_src_xyz,
        "stnxyz": all_stn_xyz,
        "npts": npts[0],
        "sample_rate": sampling_rate[0],
        "delta": delta[0],
        "distance": all_distance,
        "p_arr": all_p_arr_npts,
        "travel_time": all_travel_time,
        "xy_distance": all_xy_distance,
        "norm_A": A_norm
        }

    return sample_dict


def gen_train_h5(out_file, num_dataset, wave_folder_path, noise_path, velocity, receivers, num_event, samp_rate):
    hf_o = h5py.File(out_file, 'w')

    for i in range(num_dataset):
        sample_list = get_1sampdata(
            wave_folder_path=wave_folder_path,
            noise_path=noise_path,
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
        print('idx', i, 'srcxyz:', sample_list["srcxyz"][0])
        # print('idx', i, 'srcxyz:', sample_list["srcxyz"], 'stnxyz', sample_list["stnxyz"],
        #       'npts:', sample_list["npts"], 'sample_rate:', sample_list["sample_rate"],
        #       'delta:', sample_list["delta"], 'distance:', sample_list["distance"],
        #       'p_arr:', sample_list["p_arr"], 'travel_time:', sample_list["travel_time"]
        #       )
    hf_o.close()


# 参数设置
velocity = 4500 * 1000  # mm/s
num_receivers = 32
num_event = 39
new_sampling_rate = 3000000
wave_folder_path = r"E:\lgx\code\AEMNet\locate\iog_data\iog_train_highSNR_basewave_decay"
noise_path = r"E:\lgx\code\AEMNet\locate\iog_data\noise"
out_file = r'E:\lgx\code\AEMNet\locate\iog_data/train_dataset_xy1_300000_decay_0.1.h5'
# 重采样
# for idx, file_path in enumerate([wave_folder_path for i in range(10)]):
#     sac_files = [f"{file_path}/idx{idx}_{j}.sac" for j in range(6)]
#     for sac_path in sac_files:
#         resampling_rate_sac(sac_path, new_sampling_rate, overwrite=True)

gen_train_h5(out_file=out_file,
             num_dataset=300000,
             wave_folder_path=wave_folder_path,
             noise_path=noise_path,
             velocity=velocity,
             receivers=RECEIVERS,
             num_event=num_event,
             samp_rate=new_sampling_rate)







