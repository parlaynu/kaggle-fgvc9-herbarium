from PySide6.QtCore import QObject

from PIL import Image

from herbarium.nodes import data


class DataSource(QObject):
    
    def __init__(self, dsroot, parent=None):
        super().__init__(parent)
        
        self._dsroot = dsroot

        self._tdata = data.HerbariumDataset(dsroot, "train", 16, shuffle=False, load_images=False, excludes=False)
        self._vdata = data.HerbariumDataset(dsroot, "val", 16, shuffle=False, load_images=False, excludes=False)
        #self._tdata = self._vdata

        self._splits = ["train", "val"]
        self._split = "train"
        
        self._category_ids = set()
        for item in self._vdata:
            self._category_ids.add(item['category_id'])
        self._max_category = max(self._category_ids)
        self._num_categories = len(self._category_ids)
        self._category_id = -1
        
        self._images = []
        self._update()
    
    def num_categories(self):
        return self._num_categories
    
    def category_id(self):
        return self._category_id
    
    def category_name(self):
        if self._category_id == -1:
            return "all"
        return self._tdata.category(self._category_id)['label']
        
    def set_category_id(self, category_id, *, up=True):
        # make sure we're setting a category_id that exists
        if category_id >= 0 and category_id not in self._category_ids:
            if up is True:
                while category_id < self._max_category:
                    if category_id in self._categories:
                        break
                    category_id += 1
            else:
                while category_id >= 0:
                    if category_id in self._categories:
                        break
                    category_id -= 1
        
        # run some bounds checks
        if category_id < 0:
            category_id = -1
        elif category_id > self._max_category:
            category_id = self._max_category
        
        self._category_id = category_id
        self._update()

    def splits(self):
        return self._splits.copy()
        
    def set_split(self, split):
        if split not in self._splits:
            split = self._splits[0]
        self._split = split
        self._update()
    
    def num_images(self):
        return len(self._images)

    def image_info(self, idx):
        if idx >= len(self._images):
            return None
        
        info = self._images[idx].copy()
        cat = self._vdata.category(info['category_id'])
        for k, v in cat.items():
            info[k] = v
            
        return info

    def image(self, idx):
        if idx >= len(self._images):
            return None

        iinfo = self._images[idx]
        pil = Image.open(iinfo['image_path'])
                
        return pil
        
    def _update(self):
        dataset = self._tdata if self._split == 'train' else self._vdata
        
        images = {}
        for info in dataset:
            if self._category_id == -1 or self._category_id == info['category_id']:
                images[info['image_id']] = info
        
        keys = list(images.keys())
        keys.sort()
        
        self._images = [images[k] for k in keys]


def quick_test():
    ds = DataSource("~/Projects/datasets/fgvc9-herbarium-2022")
    
    num_categories = ds.num_categories()
    print(f"num categories: {num_categories}")
    print(f"category name: {ds.category_name()}")
    
    for i in range(10):
        ds.set_category_id(i)
        print(f"- category name: {i} {ds.category_name()}")
    ds.set_category_id(-1)
    
    print(f"- category name: {ds.category_name()}")
    ds.set_split("train")
    print(f" - num train images: {ds.num_images()}")
    ds.set_split("val")
    print(f" - num vdate images: {ds.num_images()}")

    ds.set_category_id(12)
    print(f"- category name: {ds.category_name()}")
    ds.set_split("train")
    print(f" - num train images: {ds.num_images()}")
    ds.set_split("val")
    print(f" - num vdate images: {ds.num_images()}")
    
    iinfo = ds.image_info(0)
    print(f"- opening image {iinfo['image_name']}")
    image = ds.image(0)


if __name__ == "__main__":
    quick_test()

