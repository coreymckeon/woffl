import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


# updated methods for inputs to be md, then vd, then hd...keep it same..?
# need to include the jetpump measured depth in the WellProfile
# maybe segregate it here? Have a tubing length? Have a sand length?
class WellProfile:
    def __init__(self, md_list: list, vd_list: list) -> None:
        """Initialize a Well Profile

        Args:
            md_list (list): List of measured depths
            vd_list (list): List of vertical depths

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

    def md_interp(self, vd_dpth: float) -> float:
        """Measured Depth Interpolation

        Args:
            vd_dpth (float): Vertical Depth, feet

        Returns:
            md_dpth (float): Measured Depth, feet
        """
        return self._depth_interp(vd_dpth, self.vd_ray, self.md_ray)

    def plot_raw(self) -> None:
        """Plot the Raw Profile Data"""
        self._profileplot(self.hd_ray, self.vd_ray, self.md_ray)
        return None

    def plot_filter(self) -> None:
        """Plot the Filtered Data"""
        self._profileplot(self.hd_fit, self.vd_fit, self.md_fit)
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
        # if you want to use hd_ray and vd_ray, need a way to deal with pure vertical
        # section of wellbore and pure horizontal section of well bore on interps
        idx = np.searchsorted(self.md_ray, md_fit)
        hd_fit = self.hd_ray[idx]
        return hd_fit, vd_fit, md_fit

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
    def _profileplot(hd_ray: np.ndarray, vd_ray: np.ndarray, md_ray: np.ndarray) -> None:
        """Create a Well Profile Plot

        Annotate the graph will a label of the measured depth every 1000 feet of md.

        Args:
            hd_ray (np array): Horizontal distance, feet
            td_ray (np array): Vertical depth, feet
            md_ray (np arary): Measured depth, feet
        """
        if len(md_ray) > 20:
            plt.scatter(hd_ray, vd_ray)
        else:
            plt.plot(hd_ray, vd_ray, marker="o", linestyle="--")
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
                text=f"{int(md_ray[idx])} ft.", xy=(hd_ray[idx] + 5, vd_ray[idx] - 10), rotation=30  # type: ignore
            )
        plt.show()


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
