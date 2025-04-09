import os
import random


class imgSelect():
    '''
    图片二级目录结构：
    dir_root_path
    ├── 01
    │   ├── img_01.jpg
    │   ├── img_02.jpg
    │   ├── img_03.jpg
    │   ├── img_04.jpg
    ├── 02
    │   ├── img_01.jpg
    │   ├── img_02.jpg
    │   ├── img_03.jpg
    │   ├── img_04.jpg

    args:
        img_dir_root_path: 图片根路径
        extract_num:  要抽取的图片数量，也就是想要一次展示图片的数量
    '''

    def __init__(self, img_dir_root_path, extract_num=9):
        self.root_path = img_dir_root_path  # 图片的根路径，应该是包含二级目录
        self.extract_num = extract_num  # 要抽取的数量,适用于偏好选择窗口
        self.subdirs = self.list_subdirectories()  # 获取根路径下的所有类型的图片目录
        self.subdirs_id = [i for i in range(len(self.subdirs))]  # 文件夹对应的id
        # 每个文件夹对应的权重，权重越大，从中抽取图片的概率越大
        self.select_hobby = False  # 是否进行了图片偏好性测试
        self.subdirs_weight = [1 for i in self.subdirs]
        self.num_of_calls = 0  # 调用次数，随着调用次数来更新图片
        self.num_of_choice = [0 for i in self.subdirs]  # 用户选取每个文件夹的次数，初始为0
        self.subfiles = self.list_subfiles()  # 获取每个目录下的所有图片路径，顺序一一对应
        assert (len(self.subfiles) == len(self.subdirs))

    def list_subdirectories(self):
        '''根据根路径获取一级目录'''
        result = []
        contents = os.listdir(self.root_path)
        for item in contents:
            # 构建完整的路径
            item_path = os.path.join(self.root_path, item)
            # 判断是否为目录
            if os.path.isdir(item_path):
                # print(item)
                result.append(item_path)
        return result

    def list_files(self, dir):
        '''获取对应目录下所有图片文件的路径'''
        result = []
        contents = os.listdir(dir)
        for item in contents:
            # 构建完整的路径
            item_path = os.path.join(dir, item)
            # 判断是否为目录
            if os.path.isfile(item_path):
                # print(item)
                result.append(item_path)
        return result

    def list_subfiles(self):
        """获取所有的图片文件的路径"""
        subfiles = []
        for dir_path in self.subdirs:
            subfiles.append(self.list_files(dir_path))
        return subfiles

    def half_above_threshold(self, source, threshold_val, above_num):
        """
            根据权重的大小或者选择文件类型的次数来决定是否重置
            如果所有权重中大于参数threshold_weight的权重个数
            超过了参数threshold_num, 就返回True
        """
        count_above_threshold = 0
        # 计算大于阈值的元素个数
        for val in source:
            if val > threshold_val:
                count_above_threshold += 1

        # 判断是否超过一半
        if count_above_threshold > above_num:
            return True
        else:
            return False

    def update_weight_method_01(self, idx):
        """
            采取数值累加方法来更新权重。
            权重的初始值都为1, 用户每选择一次,
            对应文件夹权重+1, 一个文件夹的最
            高权重占比不会超过0.8
        """
        # 计算权重的最大值，所有权重的总和 * 0.8, 数值会变动
        MAX_NUM = int(sum(self.subdirs_weight) * 0.8)
        self.subdirs_weight[idx] += 1

    def update_weight_method_02(self, idx):
        """
            采取更快的方式实现权重增加，
            采用累积选取次数平方的方法来更新权重，
            依旧保证最高权重占比不会超过0.8
        """
        self.subdirs_weight = [50 for i in self.subdirs]
        for i in range(len(self.subdirs_weight)):
            self.subdirs_weight[i] += self.num_of_choice[i]

    def update_weight(self, idx):
        """
            根据传回来的索引更新权重
            更新的方式：
                1. 查看调用的次数
                2. 权重的取值范围
                3. 权重的增长方式
            args:
                idx: 用户当前选取的图片对应的索引
        """
        # 记录用户的累积选择次数
        self.num_of_choice[idx] += 1
        # 设定在前三次调用时不更新权重（依然记录）
        if self.num_of_calls <= 3:
            return
        # 重置权重，加入用户的爱好不定，在多个路径的权重
        # 都大于某个值时，对权重进行重置
        if self.half_above_threshold(self.num_of_choice, 5, int(self.extract_num / 2)):
            self.clear_weight()
            # 根据用户的累计选择次数，更新权重
        self.update_weight_method_01(idx)

    def clear_weight(self):
        """
            清除权重
        """
        self.select_hobby = False
        self.num_of_calls = 0
        self.subdirs_weight = [1 for i in self.subdirs]
        self.num_of_choice = [0 for i in self.subdirs]  # 用户选取每个文件夹的次数，初始为0

    def get_counter(self):
        """
            获取计数次数
        """
        return self.num_of_calls

    def get_img_path(self):
        """
            根据每个文件夹的权重从新生成随机数
            根据随机数来随机选取图片
            返回值为：
                1. N张(N=extract_num)张图片
                2. 图片对应所属文件夹的索引
        """
        self.num_of_calls += 1  # 调用次数加一
        # 归一化权重
        weight_sum = sum(self.subdirs_weight)
        weights_normal = [(weight / weight_sum)
                          for weight in self.subdirs_weight]
        # 根据权重，随机N个文件夹，可以重复选取
        result_idx = random.choices(
            population=self.subdirs_id,
            weights=weights_normal,
            k=self.extract_num
        )
        # 根据图片文件夹的索引，再从中随机选取图片
        result = []
        for idx in result_idx:
            img_path = random.choices(population=self.subfiles[idx])
            while (img_path in result):
                # 重复选取处理
                img_path = random.choices(population=self.subfiles[idx])
            result.append(img_path)
        return result, result_idx

    def find_max_index(self, array):
        """
            找出一个数组中的最大值，这里用来查找被选中最多次的图片类型
        """
        if not array:
            return -1  # 处理空数组情况，这里返回-1表示未找到
        max_value = array[0]
        max_index = 0
        for i in range(1, len(array)):
            if array[i] > max_value:
                max_value = array[i]
                max_index = i

        return max_index

    def get_one_img(self, use_weight=False):
        """
            获取一张图片，有两种方式：
            1. 根据权重来随机获取图片,小概率为其他类型图片,use_weight=True
            2. 根据权重占比来获取图片,只从权重最大的类型中取出一张
        """
        self.num_of_calls += 1
        if self.select_hobby:
            # 调用了偏好选择窗口
            if use_weight:
                # 归一化权重
                weight_sum = sum(self.subdirs_weight)
                weights_normal = [(weight / weight_sum)
                                  for weight in self.subdirs_weight]
                # 根据权重，随机1个文件夹，可以重复选取
                folder_idx = random.choices(
                    population=self.subdirs_id,
                    weights=weights_normal,
                    k=1
                )
                folder_idx = folder_idx[0]
                img_path = random.choices(
                    population=self.subfiles[folder_idx])
            else:
                folder_idx = self.find_max_index(self.num_of_choice)
                img_path = random.choices(population=self.subfiles[folder_idx])
        else:
            # 未调用偏好选择窗口
            weight_sum = sum(self.subdirs_weight)
            weights_normal = [(weight / weight_sum)
                              for weight in self.subdirs_weight]
            # 根据权重，随机1个文件夹，可以重复选取
            folder_idx = random.choices(
                population=self.subdirs_id,
                weights=weights_normal,
                k=1
            )
            folder_idx = folder_idx[0]
            img_path = random.choices(population=self.subfiles[folder_idx])

        return img_path[0], folder_idx

    def weight_init(self):
        """
            如果未进行用户偏好性选择, 那么初次调用时对权重初始化
            权重范围为 [1, 40], 初始化值都为20
        """
        if self.select_hobby == False and self.num_of_calls == 0:
            self.subdirs_weight = [20 for i in self.subdirs]

    def update_weight_by_time(self, second, img_id):
        """
            如果未进行图片偏好性测试, 根据用户的反应时间对权重进行校准
            反应时间小于1000ms,增加权重,反应时间1000ms - 1500ms, 权重不变,
            反应时间大于1500ms, 权重减少
        """
        if self.select_hobby == False:
            # print(F"反应时间: {second}ms")
            if second < 1000:
                self.subdirs_weight[img_id] = min(
                    40, self.subdirs_weight[img_id] + 3)
            elif second > 1500:
                self.subdirs_weight[img_id] = max(
                    1, self.subdirs_weight[img_id] - 3)
        print(self.subdirs_weight)


if __name__ == "__main__":
    # 设置图片文件夹目录路径，相对路径或绝对路径
    root_dir = "./software/imgs_folder/"
    tmp = imgSelect(root_dir, 4)
    print(tmp.subdirs)
    print(tmp.subfiles[1])
    print(tmp.subdirs_weight)
    print(tmp.subdirs_id)

    result, result_idx = tmp.get_img_path()
    print(result)
    print(result_idx)
    print(tmp.num_of_calls)
    tmp.update_weight(3)
    print(tmp.num_of_choice)
    print(tmp.subdirs_weight)

    result, result_idx = tmp.get_img_path()
    print(result)
    print(result_idx)
    print(tmp.num_of_calls)
    tmp.update_weight(3)
    print(tmp.num_of_choice)
    print(tmp.subdirs_weight)

    result, result_idx = tmp.get_img_path()
    print(result)
    print(result_idx)
    print(tmp.num_of_calls)
    tmp.update_weight(5)
    print(tmp.num_of_choice)
    print(tmp.subdirs_weight)

    result, result_idx = tmp.get_img_path()
    print(result)
    print(result_idx)
    print(tmp.num_of_calls)
    tmp.update_weight(3)
    print(tmp.num_of_choice)
    print(tmp.subdirs_weight)
