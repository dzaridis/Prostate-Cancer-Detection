class UnitTesting:
    def __init__(self, test_dict):
        self.test_dict = test_dict
        self.faulty_dims = {}
        self.faulty_intensities = {}

    def check_dims(self, num_slices=24, dims=192):
        """
        Checks if there are dimensions and depth that mismatches with the desired ones
        Args:
            num_slices [int]: depth to check
            dims [int]:  spatial dimensions to check
        """
        for seq in self.test_dict.keys():
            ls = {pat: self.test_dict[seq][pat].shape for pat in self.test_dict[seq].keys()
                  if self.test_dict[seq][pat].shape != (num_slices,dims, dims,1)}
            self.faulty_dims.update({seq: ls})

    def check_intensity_range(self, pixel_min=0, pixel_max=1):
        """
        Checks whether pixel intensities lie within the desired min max range
        Args:
            pixel_min [int]: minimum desired pixel value
            pixel_max [int]: maximum desired pixel value
        """
        for seq in self.test_dict.keys():
            ls = {pat:{"max": self.test_dict[seq][pat].max(), "min": self.test_dict[seq][pat].min()}
                  for pat in self.test_dict[seq].keys()
                  if (self.test_dict[seq][pat].max() > pixel_max or self.test_dict[seq][pat].min() < pixel_min)}
            self.faulty_intensities.update({seq: ls})

    def get_faulty_dims(self):
        return self.faulty_dims

    def get_faulty_intensities(self):
        return self.faulty_intensities
