class ImageSplitData:
    def __init__(self, image_name: str, image_path: str, split_count: tuple, labels: dict):
        self.image_name = image_name
        self.image_path = image_path
        self.split_count = split_count
        self.labels = labels

    @property
    def split_count(self):
        return self._split_count

    @split_count.setter
    def split_count(self, value):
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("split_count 必须是一个包含两个整数的元组 (m, n)")
        if not all(isinstance(x, int) and x > 0 for x in value):
            raise ValueError("split_count 的值必须是正整数")
        self._split_count = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        if not isinstance(value, dict):
            raise ValueError("labels 必须是一个字典")
        if not all(isinstance(k, tuple) and len(k) == 2 and
                   all(isinstance(i, int) and i >= 0 for i in k) and
                   isinstance(v, int) for k, v in value.items()):
            raise ValueError("labels 字典的键必须是非负整数元组 (row, col)，值必须是整数")
        self._labels = value