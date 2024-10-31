# from https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox
"""
This class loads events from dat or npy files
"""

import os
import sys
import numpy as np
from .dat_events_tools import parse_header, stream_td_data



EV_TYPE = [('t', 'u4'), ('_', 'i4')]  # Event2D
EV_STRING = 'Event2D'


class PSEELoader(object):
    """
    PSEELoader loads a dat or npy file and stream events
    """

    def __init__(self, datfile):
        """
        ctor
        :param datfile: binary dat or npy file
        """
        self._extension = datfile.split('.')[-1]  # 调用 split('.') 将文件路径字符串分割为文件名和扩展名，并取最后一个元素作为文件扩展名
        assert self._extension in ["dat", "npy"], 'input file path = {}'.format(datfile)  # 判断文件扩展名是否合法
        self._file = open(datfile, "rb")  # 以二进制只读模式打开文件，并将返回的文件对象赋值给变量 self._file
        self._start, self.ev_type, self._ev_size, self._size = parse_header(self._file)  # 调用 parse_header 函数，并将返回值分别赋值给变量 self._start、self.ev_type、self._ev_size
        assert self._ev_size != 0  # 判断事件大小是否为 0
        self._dtype = EV_TYPE  # 将全局变量 EV_TYPE 赋值给实例变量 self._dtype,表示事件类型

        self._decode_dtype = [] # 创建一个空列表self._decode_dtype,用于存储解码后的事件数据类型
        # 遍历实例变量 self._dtype 中的每个元素，如果元素的第一个字符是下划线，则将其替换为元组 ('x', 'u2') 和 ('y', 'u2'),否则直接添加到列表中
        for dtype in self._dtype:
            if dtype[0] == '_':
                self._decode_dtype += [('x', 'u2'), ('y', 'u2'), ('p', 'u1')] # 如果事件类型为 Event2D，则添加 x、y 和 p 三个字段
            else:
                self._decode_dtype.append(dtype) # 否则，直接添加事件类型

        # size
        self._file.seek(0, os.SEEK_END)  # 将文件指针移动到文件末尾
        self._end = self._file.tell()  # 获取文件末尾的位置
        self._ev_count = (self._end - self._start) // self._ev_size  # 计算事件数量
        self.done = False  # 初始化实例变量 self.done 为 False,表示数据加载完成标志
        self._file.seek(self._start)  # 将文件指针移动到事件开始的位置
        # If the current time is t, it means that next event that will be loaded has a
        # timestamp superior or equal to t (event with timestamp exactly t is not loaded yet)
        self.current_time = 0  # 初始化当前时间
        self.duration_s = self.total_time() * 1e-6  # 计算数据集的总时间
        '''上面这段代码是一个Python类的初始化方法，它接收一个参数：datfile，这个参数应该是一个包含数据的二进制文件（.dat或.npy文件）。
           首先，它检查datfile的扩展名是否为.dat或.npy。如果不是，它会抛出一个断言错误。
           接着，它打开datfile，并读取文件的头部信息，这些信息包括起始位置、事件类型、事件大小和大小。注意，这里的"事件"可能指的是在文件中存储的数据单元。
           然后，它会解码这些信息并存储到相应的属性中。
           再之后，它会计算文件的大小（通过定位到文件的末尾并获取当前位置），并据此计算出文件中包含的事件数量。
           最后，它会将文件指针重新定位到文件的起始位置，为后续的读取操作做准备。同时，它还设置了两个时间相关的属性：current_time和duration_s。current_time表示当前加载的事件的时间戳，而duration_s表示文件的总时间（以秒为单位）。
           总的来说，这个类似乎是用于读取和处理二进制数据文件的工具。
        '''

    def reset(self):
        """reset at beginning of file"""
        self._file.seek(self._start) # 将文件指针重新定位到文件的起始位置
        self.done = False # 将实例变量 self.done 重置为 False,表示数据加载完成标志
        self.current_time = 0

    def event_count(self):
        """
        event_count 方法没有参数，并返回了对象的 _ev_count 属性值。
        Getter 方法的作用是封装类的属性，使其只能通过该类提供的接口来访问和修改属性值，从而保证了数据的安全性和完整性。
        getter on event_count
        :return:
        """
        return self._ev_count

    def get_size(self):
        """"(height, width) of the imager might be (None, None)"""
        return self._size

    def __repr__(self):
        """
        打印属性
        :return:
        """
        wrd = ''  # 初始化一个空字符串
        wrd += 'PSEELoader:' + '\n'  # 在字符串末尾添加 "PSEELoader:" 并换行
        wrd += '-----------' + '\n'  # 在字符串末尾添加 "-----------" 并换行
        if self._extension == 'dat':  # 如果文件扩展名为 "dat"
            wrd += 'Event Type: ' + str(EV_STRING) + '\n'  # 在字符串末尾添加 "Event Type: EV_STRING" 并换行
        elif self._extension == 'npy':  # 如果文件扩展名为 "npy"
            wrd += 'Event Type: numpy array element\n'  # 在字符串末尾添加 "Event Type: numpy array element" 并换行
        wrd += 'Event Size: ' + str(self._ev_size) + ' bytes\n'  # 在字符串末尾添加 "Event Size: 事件大小字节数" 并换行
        wrd += 'Event Count: ' + str(self._ev_count) + '\n'  # 在字符串末尾添加 "Event Count: 事件数量" 并换行
        wrd += 'Duration: ' + str(self.duration_s) + ' s \n'  # 在字符串末尾添加 "Duration: 持续时间秒" 并换行
        wrd += '-----------' + '\n'  # 在字符串末尾添加 "-----------" 并换行
        return wrd  # 返回拼接好的字符串
    

    def load_n_events(self, ev_count):
        """
        加载一批事件
        :param ev_count: 将要加载的事件数量
        :return: 事件
        注意当前时间将递增以达到尚未加载的第一个事件的时间戳
        """
        # 创建一个空的事件缓冲区
        event_buffer = np.empty((ev_count + 1,), dtype=self._decode_dtype)
    
        # 获取当前文件指针位置
        pos = self._file.tell()
        # 计算剩余事件数量
        count = (self._end - pos) // self._ev_size
        # 如果需要加载的事件数量大于等于剩余事件数量
        if ev_count >= count:
            # 标记已完成
            self.done = True
            # 更新需要加载的事件数量
            ev_count = count
            # 从文件中读取事件数据到缓冲区
            stream_td_data(self._file, event_buffer, self._dtype, ev_count)
            # 更新当前时间为第一个未加载事件的时间戳加1
            self.current_time = event_buffer['t'][ev_count - 1] + 1
        else:
            # 从文件中读取事件数据到缓冲区
            stream_td_data(self._file, event_buffer, self._dtype, ev_count + 1)
            # 更新当前时间为第一个未加载事件的时间戳
            self.current_time = event_buffer['t'][ev_count]
            # 将文件指针移动到下一个事件的起始位置
            self._file.seek(pos + ev_count * self._ev_size)
    
        # 返回已加载的事件数据
        return event_buffer[:ev_count]
    

    def load_delta_t(self, delta_t): # 加载delta_t时间内的数据
        """
        从文件中加载指定时间段内的事件
        :param delta_t: 时间段的长度（微秒）
        :return: 事件数据
        注意：当前时间会增加 delta_t
        """
        if delta_t < 1:
            raise ValueError("load_delta_t(): delta_t must be at least 1 micro-second: {}".format(delta_t))

        if self.done or (self._file.tell() >= self._end): # .tell() 是 Python 中的一个方法，用于获取文件指针当前的位置。在处理文件时，我们经常需要知道当前读取到哪里，这时就可以使用 .tell() 方法来获取当前的文件指针位置。
            self.done = True
            return np.empty((0,), dtype=self._decode_dtype)

        final_time = self.current_time + delta_t  # 计算结束时间
        tmp_time = self.current_time  # 初始化临时时间
        start = self._file.tell()  # 获取文件指针位置
        pos = start  # 初始化文件指针位置
        nevs = 0  # 初始化事件数量
        batch = 100000  # 每次读取的事件数量
        event_buffer = []  # 初始化事件缓冲区
        # 当读取到足够的事件或文件结束时停止读取数据 data is read by buffers until enough events are read or until the end of the file
        while tmp_time < final_time and pos < self._end:  # 循环读取事件数据，直到读取到足够的事件或者到达文件末尾
            count = (min(self._end, pos + batch * self._ev_size) - pos) // self._ev_size  # 计算要读取的事件数量
            buffer = np.empty((count,), dtype=self._decode_dtype)  # 创建一个事件缓冲区
            stream_td_data(self._file, buffer, self._dtype, count)  # 从文件中读取事件数据
            tmp_time = buffer['t'][-1]  # 更新临时时间
            event_buffer.append(buffer)  # 将读取到的事件保存到事件缓冲区中
            nevs += count  # 更新事件数量
            pos = self._file.tell()  # 更新文件指针位置
        if tmp_time >= final_time:  # 如果读取到了足够的事件
            self.current_time = final_time  # 更新当前时间
        else:  # 如果读取到了文件末尾
            self.current_time = tmp_time + 1  # 更新当前时间
        assert len(event_buffer) > 0
        idx = np.searchsorted(event_buffer[-1]['t'], final_time)  # 查找最后一个事件中结束时间的位置
        event_buffer[-1] = event_buffer[-1][:idx]  # 截取最后一个事件中结束时间之前的部分
        event_buffer = np.concatenate(event_buffer)  # 将所有事件合并成一个数组
        idx = len(event_buffer)
        self._file.seek(start + idx * self._ev_size)  # 将文件指针移动到下一个事件的位置
        self.done = self._file.tell() >= self._end  # 更新 done 标志，True或False
        return event_buffer  # 返回事件数据

    def seek_event(self, ev_count): # 用于在文件中查找指定数量的事件。函数接受一个参数ev_count,表示要查找的事件数量。
        """
        在文件中通过 ev_count 事件进行查找
        :param ev_count: 在 ev_count 事件后查找文件 :param ev_count: seek in the file after ev_count events
        注意当前时间将被设置为下一个事件的时间戳。
        """
        if ev_count <= 0:
            self._file.seek(self._start)
            self.current_time = 0
        elif ev_count >= self._ev_count:
            # 我们将光标向前移动一个事件并读取最后一个事件
            # 这将把文件光标放在正确的位置
            # current_time 将被设置为最后一个事件的时间戳 + 1
            self._file.seek(self._start + (self._ev_count - 1) * self._ev_size)
            self.current_time = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0] + 1
        else:
            # 我们将光标放在第 *ev_count* 个事件上
            self._file.seek(self._start + ev_count * self._ev_size)
            # 我们读取下一个事件的时间戳(这会改变文件中的位置)
            self.current_time = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0]
            # 这就是为什么我们在这里回到正确位置的原因
            self._file.seek(self._start + ev_count * self._ev_size)
        self.done = self._file.tell() >= self._end # 函数最后将文件指针当前位置与结束位置进行比较，如果当前位置大于等于结束位置，则将 done 属性设置为 True,表示查找完成。
    
    def seek_time(self, final_time, term_criterion: int = 100000):
        """
        二分查找算法，用于定位文件中的时间戳 seek in the file by ev_count events
        :param final_time: 目标时间戳
        :param term_criterion: 二分查找终止条件（事件数）
        该函数会将这些事件加载到缓冲区中，并使用numpy searchsorted确保结果始终准确
        Note that current time will be set to the timestamp of the next event.
        """
        # 如果目标时间戳大于总时间，则将文件指针设置为文件末尾并返回
        if final_time > self.total_time():
            self._file.seek(self._end)
            self.done = True
            self.current_time = self.total_time() + 1
            return

        # 如果目标时间戳小于等于0，则重置文件指针并返回
        if final_time <= 0:
            self.reset()
            return

        # 设置二分查找的上下界
        low = 0
        high = self._ev_count

        # 二分查找
        while high - low > term_criterion:
            middle = (low + high) // 2

            # 定位到中间事件并获取其时间戳
            self.seek_event(middle)
            mid = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0] # 由于只读取了一个数据，因此 count 参数设置为 1。['t'][0] 表示从返回的 NumPy 数组中取出索引为 0 的元素，即第一个元素。该元素是个一维数组，使用 ['t'] 来获取对应的浮点数值。最终，变量 mid 存储了从文件中读取的第一个浮点数。

            # 如果中间时间戳大于目标时间戳，则在文件的下半部分进行查找
            if mid > final_time:
                high = middle
            # 如果中间时间戳小于目标时间戳，则在文件的上半部分进行查找
            elif mid < final_time:
                low = middle + 1
            # 如果中间时间戳等于目标时间戳，则设置当前时间戳并返回
            else:
                self.current_time = final_time
                self.done = self._file.tell() >= self._end
                return

        # 现在我们知道目标时间戳在low和high之间
        # 定位到low事件并将事件读入缓冲区
        self.seek_event(low)
        final_buffer = np.fromfile(self._file, dtype=self._dtype, count=high - low)['t']
        # 使用numpy searchsorted在缓冲区中查找目标时间戳的索引
        final_index = np.searchsorted(final_buffer, final_time)

        # 定位到目标索引处的事件并设置当前时间戳
        self.seek_event(low + final_index)
        self.current_time = final_time
        self.done = self._file.tell() >= self._end
        return low + final_index

    def total_time(self): # 该函数用于获取视频的总时长(单位为毫秒),前提是没有溢出
        """
        获取视频的总时长(单位为mus),前提是没有溢出
        :return:
        """
        if not self._ev_count:
            return 0
        # 保存类的状态
        pos = self._file.tell()
        current_time = self.current_time
        done = self.done
        # 读取最后一个事件的时间戳
        self.seek_event(self._ev_count - 1)
        time = np.fromfile(self._file, dtype=self._dtype, count=1)['t'][0]
        # 恢复类的状态
        self._file.seek(pos)
        self.current_time = current_time
        self.done = done
    
        return time
    
    def __del__(self):
        self._file.close()

