import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


def profileplot(hd_array, tvd_array, md_array):
    """Create a Well Profile Plot

    Args:
        hd_array (numpy array): horizontal distance array
        td_array (numpy array): vertical depth array
        md_array (numpy arary): measured depth, annotated
    """
    plt.scatter(hd_array, tvd_array)
    plt.gca().invert_yaxis()
    plt.title(f"Dir Survey, Length: {max(md_array)} ft")
    plt.xlabel("Horizontal Distance, Feet")
    plt.ylabel("True Vertical Depth, Feet")

    # find the position in the measured depth array closest to every 1000'
    md_match = np.arange(1000, max(md_array), 1000)
    idxs = np.searchsorted(md_array, md_match)

    # find the angle of the line near each point so you can put the text
    # perpendicular to the line being graphed...do later?

    # annotate approximately every 1000' of measured depth
    for idx in idxs:
        plt.annotate(text=f"{int(md_array[idx])} ft.", xy=(hd_array[idx] + 5, tvd_array[idx] - 10), rotation=30)
    plt.show()


def hd_array(md_array, tvd_array):
    """Horizontal Distance Array

    Calculate a horizontal distance array. Which can be graphed with the
    vertical depth array to give an accurate visualization of the well profile.
    The horizontal distance is how far the well bore has travelled horizontally
    away from the wellhead.

    Args:
        md_array (numpy array): measured depth array
        tvd_array (numpy array): true vertical depth array

    Returns:
        hd_array (numpy array): horizontal distance array
    """
    c1 = md_array[:-1]  # top to second to last
    c2 = md_array[1:]  # second value down
    c = c2 - c1

    b1 = tvd_array[:-1]
    b2 = tvd_array[1:]
    b = b2 - b1

    # add something where if the value is funky or can't be solved for
    # you just put in a zero for the horizontal distance there.

    # append a zero to the top to make the array match original sizes
    a = np.zeros(1)
    a = np.append(a, np.sqrt(c**2 - b**2))
    # perform a rolling sum, since the previous values are finite differences
    a = np.cumsum(a)
    return a


def vertical_angle(md_array, tvd_array):
    """Vertical Angle, degrees

    Imagine a triangle whose hypotenuse starts at (0, 0) and ends at (x1, y1).
    The angle being calculated is between the hypotenuse and y-axis.

    Args:
        md_array (numpy array): measured depth, triangle hypotenuse
        tvd_array (numpy array): vertical depth, triangle cozy side

    Returns:
        a_array (numpy array): angle array, same size as input arrays
    """

    md1 = md_array[:-1]  # top to second to last
    md2 = md_array[1:]  # second value down

    vd1 = tvd_array[:-1]
    vd2 = tvd_array[1:]

    x = (vd2 - vd1) / (md2 - md1)

    theta = np.arccos(x)  # angle in radians
    theta = np.degrees(theta)  # convert to degrees
    theta = np.append(np.zeros(1), theta)  # place a zero at the top
    theta = np.around(theta, decimals=1)  # round it out!

    return theta


def segments_fit(X, Y, maxcount=18):
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


class WellProfile:
    def __init__(self, md_list: list, tvd_list: list) -> None:
        """Initialize a Well Profile

        Args:
            md_list (list): List of measured depths
            tvd_list (list): List of vertical depths

        Returns:
            Self
        """
        if len(md_list) != len(tvd_list):
            raise ValueError("Lists for Measured Depth and Vertical Depth need to be the same length")

        if max(md_list) < max(tvd_list):
            raise ValueError("Measured Depth needs to extend farther than Vertical Depth")

        md_array = np.array(md_list)
        tvd_array = np.array(tvd_list)

        self.md_array = md_array
        self.tvd_array = tvd_array

    def __repr__(self):
        final_md = round(self.md_array[-1], 0)
        final_tvd = round(self.tvd_array[-1], 0)

        return f"Profile is {final_md} ft. long and {final_tvd} ft. deep"

    def tvd_interp(self, md_dpth: float) -> float:
        """True Vertical Depth Interpolation

        Args:
            md_dpth (float): Measured Depth Point, feet

        Returns:
            tvd_dpth (float): Vertical Depth Point, Feet
        """
        if (min(self.md_array) < md_dpth < max(self.md_array)) is False:
            raise ValueError(f"{md_dpth} feet is not inside survey boundary")

        tvd_dpth = np.interp(md_dpth, self.md_array, self.tvd_array)
        return round(tvd_dpth, 1)

    def md_interp(self, tvd_dpth):
        """Measured Depth Interpolation

        Args:
            tvd_dpth (float): Vertical Depth Point, feet

        Returns:
            md_dpth (float): Measured Depth Point, feet
        """
        if (min(self.tvd_array) < tvd_dpth < max(self.tvd_array)) is False:
            raise ValueError(f"{tvd_dpth} feet is not inside survey boundary")

        md_dpth = np.interp(tvd_dpth, self.tvd_array, self.md_array)
        return round(md_dpth, 1)

    def filter(self):
        """Filter WellProfile to the Minimal Data Points

        Feed the method the raw measured depth and raw true vertical depth data.
        Method will assess the raw data and represent it in the smallest number of data points.
        Method uses the segments fit function from above.

        Args:

        Returns:
            md_fit (np.array): Measured Depth Filtered Data
            tvd_fit (np.array): Vertical Depth Filter Data

        """
        md_arr = self.md_array
        tvd_arr = self.tvd_array

        hd_arr = hd_array(md_arr, tvd_arr)

        hd_fit, tvd_fit = segments_fit(hd_arr, tvd_arr)

        # find the indicies that the returned data fit on the old array
        # use tvd while the well is vertical
        # use hd while the well is horizontal
        # could use the angle to make this judgement on
        # this judgement as well for picking what to interp on...
        idx1 = np.searchsorted(tvd_arr, tvd_fit[:2])  # first two values should be sorted on tvd
        idx2 = np.searchsorted(hd_arr, hd_fit[2:])  # once well is no longer vertical, use hd
        idxs = np.append(idx1, idx2)
        md_fit = md_arr[idxs]

        self.md_fit = md_fit
        self.tvd_fit = tvd_fit

        return md_fit, tvd_fit

    def vert_angle(self):
        """Calculate vertical angle of the points

        Uses the filtered data, but raw data could also be used.
        """
        md_fit, tvd_fit = self.filter()
        angle = vertical_angle(md_fit, tvd_fit)
        return angle

    def plot_raw(self) -> None:
        """Plot the Raw Profile Data"""
        hd_arr = hd_array(self.md_array, self.tvd_array)
        profileplot(hd_arr, self.tvd_array, self.md_array)
        return None

    def plot_filter(self) -> None:
        """Plot the Filtered Data"""
        md_fit, tvd_fit = self.filter()  # run the method
        # profileplot(md_fit, tvd_fit)
        return None
