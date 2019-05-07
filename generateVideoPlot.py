import csv
import numpy as np
import matplotlib.pyplot as plt

filePath = 'gpu_video_data.csv'
with open(filePath, 'r') as f:
    header = f.readline().strip('\n').split(',')
    reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
    frame_times = np.asarray(list(reader))

gpu_frame_times = frame_times[:, 1]
cpu_frame_times = frame_times[:, 2]
x = np.arange(cpu_frame_times.size)
print(cpu_frame_times.size)

data = [{'frame_size': 100, 'full': 0.31969428062438965, 'fbf': 0.3114511966705322}, {'frame_size': 500, 'full': 1.3878278732299805, 'fbf': 1.5456387996673584}, {'frame_size': 1000, 'full': 2.7657015323638916, 'fbf': 3.092517375946045}, {'frame_size': 1600, 'full': 4.625614881515503, 'fbf': 4.970510244369507}]

data_full = [d['full'] for d in data]
data_fbf = [d['fbf'] for d in data]
data_label = [d['frame_size'] for d in data]
N = len(data_full)
plt.rcParams.update({'font.size': 12})
plt.figure("video_frame_by_frame")
# plt.title("CPU vs GPU performance of Gaussian Blur for a video frame by frame")
# plt.plot(x, gpu_frame_times)
# plt.plot(x, cpu_frame_times)
# plt.legend(['GPU Time', 'CPU Time', '3'])
# plt.xlabel('Frame Count')
# plt.ylabel('Elapsed Time(s)')
# plt.show()
ind = np.arange(N)
width = 0.25
plt.bar(ind, data_full, width, color='#D83427', label='Full Video')
plt.bar(ind + width, data_fbf, width, label='Frame by Frame')

plt.ylabel('Time(s)')
plt.xlabel('Number of Frames')
plt.title('Frame by Frame vs Full Video Performance for Gaussian Blur')

plt.xticks(ind + width / 2, data_label)
plt.legend(loc='best')
plt.show()
