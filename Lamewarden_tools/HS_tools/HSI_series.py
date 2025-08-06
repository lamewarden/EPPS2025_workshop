from Lamewarden_tools.HS_tools.readHS import *


class HSI_series:
    def __init__(self):
        self.content = {}
        
    def __str__(self):
        return str(self.content.keys())

    def upload(self, path_to_dir, min_rows = 1, format='hdr', replace_existing = True, filter="Calibration.hdr"):
        all_images = {}
        for root, dirs, files in os.walk(path_to_dir):
            for file in files:
                filepath = os.path.join(root, file)
                # select only files with .hdr extension and with more than 15 rows
                if filepath.endswith('.hdr') and int(HS_image(filepath).rows) > min_rows and filter not in file:
                    if format == 'hdr':
                        img = HS_image(filepath)
                    else:
                        img = MS_image(filepath)
                    all_images[img.name] = img

        if replace_existing:
            self.content = all_images
        else:
            self.content.update(all_images)

    def __getitem__(self, key):
        return self.content.get(key, None)
    
    def set_value(self, key, value):
        self.content[key] = value

    def calibrate_all(self, dc=True):
        for value in self.content.values():
            value.calibrate(dc=dc)
            
    def normalize_all(self, to_wl = 751):
        for value in self.content.values():
            value.normalize(to_wl=to_wl)
    
    def clip_all(self, clip_tuple=(100,-100), axis=1):
        for value in self.content.values():
            if axis == 1:
                value.img = value.img[:, clip_tuple[0]:clip_tuple[1],:]
                value.rgb_sample = value.rgb_sample[:, clip_tuple[0]:clip_tuple[1],:]
            elif axis == 0:
                value.img = value.img[clip_tuple[0]:clip_tuple[1], :,:]
                value.rgb_sample = value.rgb_sample[clip_tuple[0]:clip_tuple[1], :,:]
            elif axis == 2:
                value.img = value.img[:,:,clip_tuple[0]:clip_tuple[1]]
            else:
                raise ValueError("Axis must be 0 (rows) or 1 (columns)")
            
    def __iter__(self):
        return iter(self.content.values())

    @classmethod
    # alternative constructor
    def from_directory(cls, path_to_dir, min_rows=1, format='hdr', replace_existing=True, filter="Calibration.hdr"):
        instance = cls()
        instance.upload(path_to_dir, min_rows, format, replace_existing, filter)
        return instance

    

