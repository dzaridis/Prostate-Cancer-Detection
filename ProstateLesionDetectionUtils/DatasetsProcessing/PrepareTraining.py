import ProstateLesionDetectionUtils


class PrepareTrain:
    def __init__(self, pic, p158, prostatenet):
        """

        Args:
            pic[dict]  : keys are the sequences ["T2", "ADC", "DWI", "Lesions, "WG"] and values are the dictionaries
            with patient names and values the 3D array
            p158[dict] : keys are the sequences ["T2", "ADC", "DWI", "Lesions, "WG"] and values are the dictionaries
            with patient names and values the 3D array
        """
        self.pic = pic
        self.p158 = p158
        self.prostatenet = prostatenet

    def prepare_158(self):
        """
        Returns: dictionary with prostate 158 patients with the structure it has been given into the class.
        it keeps patients with only lesion, it fixes/standardizes te intensities and  converts lesions masks
        to be binary
        """
        for seq in ["T2","ADC","DWI"]:
            pct2 = ProstateLesionDetectionUtils.DatasetsProcessing.PreprocessingUtils3D.\
                PrepareDatasetForTraining(self.p158, key=seq)
            pct2.keep_lesion_patients()
            pct2.fix_intensities()
            pct2.convert_mask_to_binary()
            self.p158 = pct2.get_processed_patients()
        return self.p158

    def prepare_pic(self):
        """
        Returns: dictionary with prostate 158 patients with the structure it has been given into the class.
        it keeps patients with only lesion, it fixes/standardizes te intensities and  converts lesions masks
        to be binary
        """
        for seq in ["T2","ADC","DWI"]:
            pct2 = ProstateLesionDetectionUtils.DatasetsProcessing.PreprocessingUtils3D.\
                PrepareDatasetForTraining(self.pic, key=seq)
            pct2.keep_lesion_patients()
            pct2.fix_intensities()
            pct2.convert_mask_to_binary()
            self.pic = pct2.get_processed_patients()
        return self.pic

    def prepare_prostatenet(self):
        """
        Returns: dictionary with prostate net patients with the structure it has been given into the class.
        it keeps patients with only lesion, it fixes/standardizes the intensities and  converts lesions masks
        to be binary
        """
        for seq in ["T2", "ADC", "DWI"]:
            t2_proc = ProstateLesionDetectionUtils.DatasetsProcessing.PreprocessingUtils3D.MRIProcessing(self.prostatenet[seq])
            t2_proc.Normalization8bit()
            t2_proc.SampleWiseStandardization()
            self.prostatenet[seq] = t2_proc.GetPatient()
        return self.prostatenet

    def create_zones_158(self):
        """

        Returns:dictionary with the XY and Z keys which indicate the zones created for the metric

        """
        metr =ProstateLesionDetectionUtils.DetectionMetrics.ModelDetoriention.DetorientLesions(self.p158["T2"],
                                                                                               self.p158["Lesions"],
                                                                                               rates=[10, 20, 30])
        metr.CreateZones()
        metr.KeepZonesNonZeroImageValues()
        metr.CreateZaxisZone()
        P158LesionZones = metr.GetZones()
        P158Zlesions = metr.GetZaxisZones()
        return {"XY": P158LesionZones,"Z": P158Zlesions}

    def create_zones_pic(self):
        """

        Returns:dictionary with the XY and Z keys which indicate the zones created for the metric

        """
        metr = ProstateLesionDetectionUtils.DetectionMetrics.ModelDetoriention.DetorientLesions(self.pic["T2"],
                                                                                               self.pic["Lesions"],
                                                                                               rates=[10, 20, 30])
        metr.CreateZones()
        metr.KeepZonesNonZeroImageValues()
        metr.CreateZaxisZone()
        PicLesionZones = metr.GetZones()
        PicZlesions = metr.GetZaxisZones()
        return {"XY": PicLesionZones,"Z": PicZlesions}