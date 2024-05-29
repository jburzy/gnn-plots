class PlotBase:
    def __init__(
        self,
        plot_config: dict | None = None
    ):
        """A base class for all plot types in this package.

        This class is responsible for containing things that are common to all
        plot types. These are:

        Parameters
        ----------
        plot_config : dict
            Dictionary (read in from yaml) containing information about the plots.
        """
        super().__init__()
        self.config = plot_config