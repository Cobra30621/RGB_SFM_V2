from pathlib import Path
import glob
import re
import torchvision

__all__ = ['gray_transform']

# 二值化
class ThresholdTransform(object):
        def __call__(self,x):
            threshold = x.mean()
            return (x>threshold).to(x.dtype)
        
gray_transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    ThresholdTransform()
])

def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path
    

