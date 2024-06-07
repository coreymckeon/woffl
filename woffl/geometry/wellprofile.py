import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


# make a dictionary eventually where you can input jetpump sleeves or gas lift mandrels
# updated methods for inputs to be md, then vd, then hd...keep it same..?
# need to include the jetpump measured depth in the WellProfile
# maybe segregate it here? Have a tubing length? Have a sand length?
class WellProfile:
    """Well Profile Class

    Create a wellprofile, which is the subsurface geometry of the measured depth versus vertical depth.
    Can be used to interpolate values for understanding how measured depth relates to vertical depth.
    """

    def __init__(self, md_list: list | np.ndarray, vd_list: list | np.ndarray, jetpump_md: float) -> None:
        """Create a Well Profile

        Args:
            md_list (list): List of measured depths
            vd_list (list): List of vertical depths
            jetpump_md (float): Measured depth of Jet Pump, feet

        Returns:
            Self
        """
        if len(md_list) != len(vd_list):
            raise ValueError("Lists for Measured Depth and Vertical Depth need to be the same length")

        if max(md_list) < max(vd_list):
            raise ValueError("Measured Depth needs to extend farther than Vertical Depth")

        self.md_ray = np.array(md_list)
        self.vd_ray = np.array(vd_list)
        self.hd_ray = self._horz_dist(self.md_ray, self.vd_ray)
        self.jetpump_md = jetpump_md

        # here is the real question, do I even need the raw data? just filter the data
        # at the __init__ and be done? Is the raw data just more stuff to wade through?
        self.hd_fit, self.vd_fit, self.md_fit = self.filter()  # run the method

    def __repr__(self):
        final_md = round(self.md_ray[-1], 0)
        final_vd = round(self.vd_ray[-1], 0)

        return f"Profile is {final_md} ft. long and {final_vd} ft. deep"

    def vd_interp(self, md_dpth: float) -> float:
        """Vertical Depth Interpolation

        Args:
            md_dpth (float): Measured Depth, feet

        Returns:
            vd_dpth (float): Vertical Depth, feet
        """
        return self._depth_interp(md_dpth, self.md_ray, self.vd_ray)

    def hd_interp(self, md_dpth: float) -> float:
        """Horizontal Distance Interpolation

        Args:
            md_dpth (float): Measured Depth, feet

        Returns:
            hd_dist (float): Horizontal Distance, feet
        """
        return self._depth_interp(md_dpth, self.md_ray, self.hd_ray)

    def md_interp(self, vd_dpth: float) -> float:
        """Measured Depth Interpolation

        Args:
            vd_dpth (float): Vertical Depth, feet

        Returns:
            md_dpth (float): Measured Depth, feet
        """
        return self._depth_interp(vd_dpth, self.vd_ray, self.md_ray)

    @property
    def jetpump_vd(self) -> float:
        """Jet Pump True Vertical Depth, Feet"""
        jp_vd = self.vd_interp(self.jetpump_md)
        return jp_vd

    @property
    def jetpump_hd(self) -> float:
        """Jet Pump Horizontal Distance, Feet"""
        jp_hd = self.hd_interp(self.jetpump_md)
        return jp_hd

    # make a plot of the filtered data on top of the raw data
    def plot_raw(self) -> None:
        """Plot the Raw Profile Data"""

        self._profileplot(self.hd_ray, self.vd_ray, self.md_ray, self.jetpump_hd, self.jetpump_vd, self.jetpump_md)
        return None

    def plot_filter(self) -> None:
        """Plot the Filtered Data"""
        self._profileplot(self.hd_fit, self.vd_fit, self.md_fit, self.jetpump_hd, self.jetpump_vd, self.jetpump_md)
        return None

    def filter(self):
        """Filter WellProfile to the Minimal Data Points

        Feed the method the raw measured depth and raw true vertical depth data.
        Method will assess the raw data and represent it in the smallest number of data points.
        Method uses the segments fit function from above.

        Args:

        Returns:
            md_fit (np array): Measured Depth Filtered Data
            vd_fit (np array): Vertical Depth Filter Data
            hd_fit (np array): Horizontal Dist Filter Data
        """
        # have to use md since the valve will always be increasing
        md_fit, vd_fit = segments_fit(self.md_ray, self.vd_ray)
        md_fit[0], vd_fit[0] = 0, 0  # first values always need to start at zero
        idx = np.searchsorted(self.md_ray, md_fit)
        hd_fit = self.hd_ray[idx]
        return hd_fit, vd_fit, md_fit

    def outflow_spacing(self, seg_len: float) -> tuple[np.ndarray, np.ndarray]:
        """Outflow Piping Spacing

        Break the outflow piping into nodes that can be fed  with piping dimension
        flowrates and etc to calculate differential pressure across them. Outflow is
        assumed to start at the jetpump.

        Args:
            seg_len (float): Segment Length of Outflow Piping, feet

        Returns:
            md_seg (np array): Measured depth Broken into segments
            vd_seg (np array): Vertical depth broken into segments
        """
        return self._outflow_spacing(self.md_fit, self.vd_fit, self.jetpump_md, seg_len)

    @staticmethod
    def _outflow_spacing(
        md_fit: np.ndarray, vd_fit: np.ndarray, outflow_md: float, seg_len: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Outflow Piping Spacing

        Break the outflow piping into nodes that can be fed  with piping dimension
        flowrates and etc to calculate differential pressure across them.

        Args:
            md_fit (np array): Filtered Measured Depth Data, feet
            vd_fit (np array): Filtered Vertical Depth Data, feet
            outflow_md (float): Depth the outflow node ends, feet
            seg_len (float): Segment Length of Outflow Piping, feet

        Returns:
            md_seg (np array): Measured depth Broken into segments
            vd_seg (np array): Vertical depth broken into segments
        """
        # need to break it up
        outflow_vd = np.interp(outflow_md, md_fit, vd_fit)
        vd_fit = vd_fit[md_fit <= outflow_md]
        md_fit = md_fit[md_fit <= outflow_md]  # keep values less than outflow_md
        md_fit = np.append(md_fit, outflow_md)  # add the final outflow md to end
        vd_fit = np.append(vd_fit, outflow_vd)
        md1 = md_fit[:-1]  # everything but the last character
        md2 = md_fit[1:]  # everything but the first character
        dist = md2 - md1  # distance between points
        md_seg = np.array([])
        for i, dis in enumerate(dist):
            # force there to always be at least three (?) spaces?
            dis = max(int(np.ceil(dis / seg_len)), 3)  # evenly space out the spaces
            md_seg = np.append(md_seg, np.linspace(md1[i], md2[i], dis))  # double counting
        md_seg = np.unique(md_seg)  # get rid of weird double counts from linspace
        vd_seg = np.interp(md_seg, md_fit, vd_fit)
        return md_seg, vd_seg

    @staticmethod
    def _depth_interp(in_dpth: float, in_ray: np.ndarray, out_ray: np.ndarray) -> float:
        """Depth Interpolation

        Args:
            in_dpth (float): Known Depth, feet
            in_ray (list): Known List of Depths, feet
            out_ray (list): Unknown List of Depths, feet

        Returns:
            out_dpth (float): Unknown Depth, Feet
        """
        if (min(in_ray) < in_dpth < max(in_ray)) is False:
            raise ValueError(f"{in_dpth} feet is not inside survey boundary")

        out_dpth = np.interp(in_dpth, in_ray, out_ray)
        return float(out_dpth)

    @staticmethod
    def _horz_dist(md_ray: np.ndarray, vd_ray: np.ndarray) -> np.ndarray:
        """Horizontal Distance from Wellhead

        Args:
            md_ray (np array): Measured Depth array, feet
            vd_ray (np array): Vertical Depth array, feet

        Returns:
            hd_ray (np array): Horizontal Dist array, feet
        """
        md_diff = np.diff(md_ray, n=1)  # difference between values in array
        vd_diff = np.diff(vd_ray, n=1)  # difference between values in array
        hd_diff = np.zeros(1)  # start with zero at top to make array match original size
        hd_diff = np.append(hd_diff, np.sqrt(md_diff**2 - vd_diff**2))  # pythagorean theorem
        hd_ray = np.cumsum(hd_diff)  # rolling sum, previous values are finite differences
        return hd_ray

    @staticmethod
    def _profileplot(
        hd_ray: np.ndarray, vd_ray: np.ndarray, md_ray: np.ndarray, hd_jp: float, vd_jp: float, md_jp: float
    ) -> None:
        """Create a Well Profile Plot

        Annotate the graph will a label of the measured depth every 1000 feet of md.
        Future note, should the hd and vd axis scales be forced to match? How hard is that?

        Args:
            hd_ray (np array): Horizontal distance, feet
            td_ray (np array): Vertical depth, feet
            md_ray (np arary): Measured depth, feet
            hd_jp (float): Horizontal distance jetpump, feet
            vd_jp (float): Vertical depth jetpump, feet
            md_jp (float): Measured depth jetpump, feet
        """
        if len(md_ray) > 20:
            plt.scatter(hd_ray, vd_ray, label="Survey")
        else:
            plt.plot(hd_ray, vd_ray, marker="o", linestyle="--", label="Survey")

        # plot jetpump location
        plt.plot(hd_jp, vd_jp, marker="o", color="r", linestyle="", label=f"Jetpump MD: {int(md_jp)} ft")

        plt.gca().invert_yaxis()
        plt.title(f"Dir Survey, Length: {max(md_ray)} ft")
        plt.xlabel("Horizontal Distance, Feet")
        plt.ylabel("True Vertical Depth, Feet")

        # find the position in the measured depth array closest to every 1000'
        md_match = np.arange(1000, max(md_ray), 1000)
        idxs = np.searchsorted(md_ray, md_match)
        idxs = np.unique(idxs)  # don't repeat values, issue on filtered data

        # annotate every ~1000' of measured depth
        for idx in idxs:
            plt.annotate(
                text=f"{int(md_ray[idx])} ft", xy=(hd_ray[idx] + 5, vd_ray[idx] - 10), rotation=30  # type: ignore
            )
        plt.legend()
        plt.axis("equal")
        plt.show()

    @classmethod
    def schrader(cls):
        """Schrader Bluff Generic Well Profile

        Generic Schrader Bluff well profile based on MPE-42 geometry.

        Args:
            md_list (list): MPE-42 Measured Depth
            vd_list (list): MPE-42 Vertical Depth
            jetpump_md (float): 6693 MD, feet
        """
        e42_md = [
            0.0,
            100.0,
            165.0,
            228.0,
            291.0,
            353.0,
            416.0,
            478.0,
            540.0,
            603.0,
            668.47,
            729.76,
            792.38,
            857.25,
            920.91,
            984.6,
            1049.15,
            1113.1,
            1177.05,
            1240.07,
            1303.41,
            1367.02,
            1430.27,
            1494.48,
            1557.55,
            1621.14,
            1684.77,
            1749.15,
            1812.58,
            1874.88,
            1939.92,
            2004.37,
            2067.46,
            2130.86,
            2194.94,
            2258.7,
            2320.87,
            2385.63,
            2449.12,
            2513.13,
            2576.48,
            2639.95,
            2703.65,
            2766.87,
            2831.08,
            2894.74,
            2958.25,
            3021.82,
            3085.63,
            3148.98,
            3212.81,
            3276.35,
            3339.94,
            3403.24,
            3467.48,
            3531.39,
            3595.28,
            3659.17,
            3722.71,
            3786.56,
            3850.03,
            3913.72,
            3978.01,
            4041.17,
            4104.31,
            4168.21,
            4231.85,
            4295.51,
            4359.66,
            4423.27,
            4487.4,
            4550.98,
            4614.5,
            4677.75,
            4741.46,
            4805.42,
            4869.03,
            4932.88,
            4995.97,
            5060.18,
            5124.3,
            5187.84,
            5251.05,
            5314.67,
            5378.25,
            5442.06,
            5505.96,
            5569.28,
            5633.03,
            5697.02,
            5760.39,
            5823.75,
            5887.4,
            5951.38,
            6015.04,
            6078.96,
            6142.4,
            6206.58,
            6270.37,
            6333.91,
            6397.75,
            6459.84,
            6524.75,
            6586.71,
            6652.49,
            6715.28,
            6780.34,
            6843.97,
            6907.48,
            6971.45,
            7035.26,
            7099.07,
            7160.77,
            7226.26,
            7290.11,
            7353.89,
            7417.85,
            7481.54,
            7545.28,
            7608.7,
            7672.42,
            7736.11,
            7800.37,
            7863.49,
            7927.25,
            7988.2,
            8054.5,
            8118.07,
            8181.68,
            8245.46,
            8281.05,
            8343.89,
            8407.38,
            8470.54,
            8534.02,
            8597.81,
            8661.58,
            8725.35,
            8789.48,
            8852.84,
            8916.62,
            8980.32,
            9043.87,
            9107.23,
            9171.59,
            9234.96,
            9299.44,
            9362.49,
            9426.1,
            9490.11,
            9553.21,
            9615.38,
            9678.84,
            9742.64,
            9808.66,
            9872.23,
            9935.41,
            9999.73,
            10063.41,
            10126.86,
            10188.82,
            10253.69,
            10316.73,
            10382.01,
            10445.74,
            10509.19,
            10573.15,
            10635.73,
            10700.83,
            10764.12,
            10828.26,
            10891.55,
            10955.51,
            11018.96,
            11082.5,
            11146.2,
            11210.22,
            11273.59,
            11337.62,
            11400.98,
            11465.32,
            11528.78,
            11592.23,
            11655.78,
            11719.52,
            11783.84,
            11847.09,
            11899.06,
            11969.0,
        ]
        e42_vd = [
            0.0,
            99.99993,
            164.9998,
            227.99935,
            290.99826,
            352.98873,
            415.90045,
            477.62711,
            539.09468,
            601.28561,
            665.62471,
            725.64553,
            786.6839,
            849.23387,
            909.72895,
            969.52142,
            1029.12834,
            1086.886,
            1142.92035,
            1195.86979,
            1246.65495,
            1294.72035,
            1340.50473,
            1385.43381,
            1428.37523,
            1471.40974,
            1513.13005,
            1553.35537,
            1591.27876,
            1626.59965,
            1661.85597,
            1696.10009,
            1730.24438,
            1765.41259,
            1800.31269,
            1834.35507,
            1867.03756,
            1901.25381,
            1934.75651,
            1967.46557,
            1998.69814,
            2029.40142,
            2060.78866,
            2093.37566,
            2126.38611,
            2157.8066,
            2188.78748,
            2219.87351,
            2251.48421,
            2283.33145,
            2316.59177,
            2349.52493,
            2381.89976,
            2414.34083,
            2446.81977,
            2478.95863,
            2511.55347,
            2545.04961,
            2578.31487,
            2611.12872,
            2643.90437,
            2676.51855,
            2708.89199,
            2740.8343,
            2773.53295,
            2806.82566,
            2839.99234,
            2873.28335,
            2906.88292,
            2939.94386,
            2972.93521,
            3005.71975,
            3039.41617,
            3073.06795,
            3105.89559,
            3138.52615,
            3171.34953,
            3204.22531,
            3236.33149,
            3269.27263,
            3302.01382,
            3334.32526,
            3365.67686,
            3396.43824,
            3427.01004,
            3457.3062,
            3487.8702,
            3519.25651,
            3552.14241,
            3585.66789,
            3618.43855,
            3651.30836,
            3684.69826,
            3718.18518,
            3751.55187,
            3785.14034,
            3818.61786,
            3852.53822,
            3886.05832,
            3919.55078,
            3952.98397,
            3984.90721,
            4017.83255,
            4047.81272,
            4078.69479,
            4106.71301,
            4132.14625,
            4153.88145,
            4172.14988,
            4185.70322,
            4195.60371,
            4203.70759,
            4210.84246,
            4218.72777,
            4226.75847,
            4234.71963,
            4242.5372,
            4250.67454,
            4258.59207,
            4265.81047,
            4273.01874,
            4280.80869,
            4290.62797,
            4302.23394,
            4313.94673,
            4324.38428,
            4334.55034,
            4342.87575,
            4348.3251,
            4351.59135,
            4353.64033,
            4357.31289,
            4360.21012,
            4361.49475,
            4360.70317,
            4358.36067,
            4355.36835,
            4352.37595,
            4349.40025,
            4345.94678,
            4341.85368,
            4337.35518,
            4333.00541,
            4329.04963,
            4325.62009,
            4322.51388,
            4319.17913,
            4315.88538,
            4311.25489,
            4305.55973,
            4300.31863,
            4295.45757,
            4290.87671,
            4286.85457,
            4283.08346,
            4279.96756,
            4277.0745,
            4274.19653,
            4271.69128,
            4269.77629,
            4268.20353,
            4266.69839,
            4265.88465,
            4264.86553,
            4263.4202,
            4261.29495,
            4258.29371,
            4255.05179,
            4251.50345,
            4247.81667,
            4243.87361,
            4240.77147,
            4239.12004,
            4238.77728,
            4238.91088,
            4238.01088,
            4236.00046,
            4233.81152,
            4231.047,
            4227.86384,
            4224.07663,
            4219.49024,
            4214.94324,
            4210.66554,
            4206.88594,
            4203.83414,
            4200.62917,
            4197.74213,
            4193.97265,
        ]
        return cls(md_list=e42_md, vd_list=e42_vd, jetpump_md=6693)

    @classmethod
    def kuparuk(cls):
        """Kuparuk Generic Well Profile

        Generic Kuparuk well profile based on MPC-23 geometry.
        MPC-23 is a slant Kuparuk Well, so not a perfect canidate.

        Args:
            md_list (list): MPC-23 Measured Depth
            vd_list (list): MPC-23 Vertical Depth
            jetpump_md (float): 7926 MD, feet
        """
        c23_md = [
            0.0,
            175.66,
            260.85,
            350.32,
            441.59,
            532.54,
            624.99,
            716.04,
            807.23,
            897.9,
            988.21,
            1083.47,
            1178.32,
            1273.44,
            1368.95,
            1464.38,
            1559.85,
            1655.52,
            1751.03,
            1846.05,
            1941.71,
            2036.84,
            2132.29,
            2227.69,
            2323.3,
            2418.73,
            2513.76,
            2609.83,
            2705.5,
            2801.83,
            2896.87,
            2992.52,
            3088.05,
            3183.64,
            3278.42,
            3374.05,
            3470.65,
            3566.23,
            3658.51,
            3754.9,
            3851.18,
            3946.75,
            4041.49,
            4137.1,
            4232.63,
            4327.99,
            4423.29,
            4518.18,
            4613.21,
            4707.32,
            4802.54,
            4866.3,
            5003.49,
            5098.14,
            5193.33,
            5288.94,
            5384.31,
            5479.93,
            5575.62,
            5671.0,
            5766.19,
            5861.67,
            5956.6,
            6051.9,
            6147.46,
            6243.25,
            6338.61,
            6434.16,
            6528.54,
            6625.24,
            6720.47,
            6815.18,
            6911.86,
            7007.32,
            7102.32,
            7199.03,
            7294.46,
            7390.33,
            7485.55,
            7578.88,
            7674.56,
            7770.15,
            7865.87,
            7960.62,
            8055.71,
            8151.86,
            8247.47,
            8342.88,
            8438.34,
            8533.01,
            8627.09,
            8721.8,
            8816.68,
            8909.7,
            9007.82,
            9100.37,
            9195.78,
            9290.46,
            9384.5,
            9478.58,
            9573.12,
            9668.15,
            9762.66,
            9855.91,
            9947.8,
            10042.07,
            10135.4,
            10232.76,
            10325.9,
            10421.59,
            10514.22,
            10607.71,
            10702.58,
            10766.66,
            10850.0,
        ]
        c23_vd = [
            0.0,
            175.6598,
            260.84756,
            350.29797,
            441.48102,
            532.25566,
            624.46476,
            715.18241,
            805.88878,
            895.93171,
            985.44929,
            1079.54266,
            1172.94175,
            1266.78366,
            1361.41427,
            1456.24908,
            1551.35135,
            1646.84203,
            1742.28933,
            1837.2862,
            1932.92872,
            2028.03899,
            2123.47868,
            2218.87827,
            2314.48768,
            2409.91715,
            2504.94507,
            2600.93738,
            2696.24401,
            2791.7012,
            2885.20926,
            2978.55771,
            3070.94285,
            3162.42824,
            3252.00336,
            3341.11776,
            3429.75059,
            3515.98038,
            3597.86518,
            3681.6399,
            3763.01969,
            3841.4311,
            3917.64689,
            3994.24477,
            4070.73869,
            4146.47334,
            4221.00799,
            4294.75269,
            4368.97414,
            4442.03477,
            4514.73656,
            4563.13344,
            4667.34766,
            4739.33192,
            4811.95186,
            4885.1291,
            4958.5867,
            5032.82916,
            5107.84115,
            5182.34821,
            5256.09945,
            5330.25459,
            5404.08639,
            5478.11173,
            5552.20343,
            5626.25036,
            5699.62577,
            5772.67614,
            5844.24472,
            5917.00874,
            5988.15513,
            6058.80616,
            6130.86181,
            6201.67981,
            6272.69951,
            6345.52131,
            6418.07642,
            6491.75341,
            6565.71475,
            6639.14929,
            6714.52575,
            6789.85169,
            6865.59787,
            6940.79848,
            7015.96455,
            7091.58176,
            7167.07368,
            7242.83373,
            7319.00362,
            7394.99644,
            7470.79879,
            7546.89804,
            7622.87186,
            7697.18178,
            7775.31658,
            7848.44918,
            7923.2369,
            7997.44786,
            8071.27361,
            8146.57659,
            8224.70742,
            8305.27168,
            8387.33831,
            8470.10078,
            8552.42558,
            8636.69866,
            8719.6966,
            8806.05262,
            8888.98832,
            8974.83164,
            9058.20708,
            9142.3957,
            9228.15114,
            9286.31483,
            9362.01774,
        ]
        return cls(md_list=c23_md, vd_list=c23_vd, jetpump_md=7926)


def segments_fit(X: np.ndarray, Y: np.ndarray, maxcount: int = 18) -> tuple[np.ndarray, np.ndarray]:
    """Segments Fit DankOC

    Feed the method the raw data. Function will assess the raw data.
    Then return the smallest smallest number of data points to fit.

    Args:
        X (numpy array): Raw X Data to be fit
        Y (numpy array): Raw Y Data to be fit
        maxcount (int): Max number of points to represent the profile by

    Returns:
        X_fit (np.array): Filtered X Data
        Y_fit (np.array): Filtered Y Data

    References:
        - https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2a
        - https://discovery.ucl.ac.uk/id/eprint/10070516/1/AIC_BIC_Paper.pdf
        - Comment from user: dankoc
    """
    xmin = X.min()
    xmax = X.max()

    n = len(X)

    AIC_ = float("inf")
    BIC_ = float("inf")
    r_ = None

    for count in range(1, maxcount + 1):
        seg = np.full(count - 1, (xmax - xmin) / count)

        px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.1].mean() for x in px_init])

        def func(p):
            seg = p[: count - 1]
            py = p[count - 1 :]  # noqa E203
            px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
            return px, py

        def err(p):  # This is RSS / n
            px, py = func(p)
            Y2 = np.interp(X, px, py)
            return np.mean((Y - Y2) ** 2)

        r = optimize.minimize(err, x0=np.r_[seg, py_init], method="Nelder-Mead")

        # Compute AIC/ BIC.
        AIC = n * np.log10(err(r.x)) + 4 * count
        BIC = n * np.log10(err(r.x)) + 2 * count * np.log(n)

        if (BIC < BIC_) & (AIC < AIC_):  # Continue adding complexity.
            r_ = r
            AIC_ = AIC
            BIC_ = BIC
        else:  # Stop.
            count = count - 1
            break

    return func(r_.x)  # type: ignore [return the last (n-1)]
