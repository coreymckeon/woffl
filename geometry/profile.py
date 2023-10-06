import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


def profileplot(md_array, tvd_array):
    '''Create a Well Profile Plot
    '''
    plt.scatter(md_array, tvd_array, '-o')
    plt.gca().invert_yaxis()
    plt.title(f'Directional Survey')
    plt.xlabel('Measured Depth, Feet')
    plt.ylabel('True Vertical Depth, Feet')
    plt.show()


def vertical_angle(x_array, y_array):
    """Calculate vertical angle between hypotenuse and y-axis

    Imagine a triangle whose hypotenuse starts at (0, 0) and ends at (x1, y1).
    The angle being calculated is between the hypotenuse and y-axis.

    Args:
        x_array (numpy array): x values
        y_array (numpy array): y values

    Returns:
        a_array (numpy array): angle array, will be the length of the input array - 1
    """

    x1 = x_array[1:]  # second value down
    x2 = x_array[:-1]  # top to second to last

    y1 = y_array[1:]
    y2 = y_array[:-1]

    m = (x2-x1)/(y2-y1)
    theta = np.arctan(m)
    return np.degrees(theta)


class WellProfile():

    def __init__(self, md_list: list, tvd_list: list) -> None:
        '''Initialize a Well Profile

        Args:
            md_list (list): List of measured depths
            tvd_list (list): List of vertical depths

        Returns:
            Self
        '''
        if len(md_list) != len(tvd_list) == True:
            raise ValueError(
                f'Lists for Measured Depth and Vertical Depth need to be the same length')

        if max(md_list) < max(tvd_list) == True:
            raise ValueError(
                f'Measured Depth needs to extend farther than Vertical Depth')

        md_array = np.array(md_list)
        tvd_array = np.array(tvd_list)

        self.md_array = md_array
        self.tvd_array = tvd_array

    def __repr__(self):
        final_md = round(self.md_array[-1], 0)
        final_tvd = round(self.tvd_array[-1], 0)

        return f'Profile is {final_md} ft. long and {final_tvd} ft. deep'

    def tvd_interp(self, md_dpth: float) -> float:
        '''True Vertical Depth Interpolation

        Args:
            md_dpth (float): Measured Depth Point, feet

        Returns:
            tvd_dpth (float): Vertical Depth Point, Feet
        '''
        if (min(self.md_array) < md_dpth < max(self.md_array)) == False:
            raise ValueError(f'{md_dpth} feet is not inside survey boundary')

        tvd_dpth = np.interp(md_dpth, self.md_array, self.tvd_array)
        return round(tvd_dpth, 1)

    def md_interp(self, tvd_dpth):
        '''Measured Depth Interpolation

        Args:
            tvd_dpth (float): Vertical Depth Point, feet

        Returns:
            md_dpth (float): Measured Depth Point, feet
        '''
        if (min(self.tvd_array) < tvd_dpth < max(self.tvd_array)) == False:
            raise ValueError(f'{tvd_dpth} feet is not inside survey boundary')

        md_dpth = np.interp(tvd_dpth, self.tvd_array, self.md_array)
        return round(md_dpth, 1)

    def filter(self, maxcount=18):
        '''Filter WellProfile to the Minimal Data Points

        Feed the method the raw measured depth and raw true vertical depth data.
        Method will assess the raw data and represent it in the smallest number of data points.

        Args:
            maxcount (int): Max number of points to represent the profile by.

        Returns:
            md_fit (np.array): Measured Depth Filtered Data
            tvd_fit (np.array): Vertical Depth Filter Data

        References:
            - https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2a
            - https://discovery.ucl.ac.uk/id/eprint/10070516/1/AIC_BIC_Paper.pdf
            - Comment from user: dankoc
        '''

        X = self.md_array
        Y = self.tvd_array

        xmin = X.min()
        xmax = X.max()

        n = len(X)

        AIC_ = float('inf')
        BIC_ = float('inf')
        r_ = None

        for count in range(1, maxcount+1):

            seg = np.full(count - 1, (xmax - xmin) / count)

            px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
            py_init = np.array(
                [Y[np.abs(X - x) < (xmax - xmin) * 0.1].mean() for x in px_init])

            def func(p):
                seg = p[:count - 1]
                py = p[count - 1:]
                px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
                return px, py

            def err(p):  # This is RSS / n
                px, py = func(p)
                Y2 = np.interp(X, px, py)
                return np.mean((Y - Y2)**2)

            r = optimize.minimize(
                err, x0=np.r_[seg, py_init], method='Nelder-Mead')

            # Compute AIC/ BIC.
            # read paper to verify if it should be natural log or not?
            AIC = n * np.log10(err(r.x)) + 4 * count
            BIC = n * np.log10(err(r.x)) + 2 * count * np.log(n)

            if((BIC < BIC_) & (AIC < AIC_)):  # Continue adding complexity.
                r_ = r
                AIC_ = AIC
                BIC_ = BIC
            else:  # Stop.
                count = count - 1
                break

        md_fit, tvd_fit = func(r_.x)  # type: ignore
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
        '''Plot the Raw Profile Data
        '''
        profileplot(self.md_array, self.tvd_array)
        return None

    def plot_filter(self) -> None:
        '''Plot the Filtered Data
        '''
        md_fit, tvd_fit = self.filter()  # run the method
        profileplot(md_fit, tvd_fit)
        return None
