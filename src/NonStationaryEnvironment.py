from Environment import *


class NonStationaryEnvironment(Environment):
    """
    Extends the Environment class to consider the situation in which the curves related to pricing are non-stationary

    Attributes:
        t: Time
        n_phases : Number of phases
    """

    # probabilities -> prob matrix that include for each phase the mean of all arms (versione vista a lezione)
    # capire come gestire le probabilità perchè qui ho solo una categoria con varie probabilità mentre prima avevo tante categorie con solo una probabilità
    # forse mi basta fare l'override dei metodi che usano probability?
    # alternativa è di cambiare l'environmnet facendo un refactor per essere generico per ogni classe e poi fare una sottoclasse nel caso del context
    # oppure la classe context environment puo contenere tanti environment quanti sono i context
    # oppure come faccio adesso metto un dizionario con chiave la fase

    # altrimenti si puo mettere opzionale la category tipo in environment e clairvoyant
    def __init__(self, n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs, phases_duration):
        """
        Initializes the NonStationaryEnvironment class

        :param int n_prices: Number of prices
        :param dict prices: Dictionary that maps each phase to the values(price of the product) associated to the arms
        :param dict probabilities: Dictionary that maps each phase to the bernoulli probabilities associated to the arms
        :param dict bids_to_clicks: Dictionary that maps each phase to the parameters to build the function that models the number of clicks given the bid
        :param dict bids_to_cum_costs: Dictionary that maps each phase to the parameters to build the function that models the cumulative daily click cost given the bid
        :param float other_costs: Cost of the product
        :param list phases_duration: List where each element is the length of a phase
        """

        super().__init__(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)
        self.t = 0
        self.phases_duration = phases_duration
        self.n_phases = len(self.phases_duration)

    def round_pricing(self, pulled_arm, n_clicks=1):
        """
        Simulates a round in a pricing scenario, returning the realization of the pulled price considering the phase
        of the environment

        :param int pulled_arm: Arm pulled in the current time step
        :param int n_clicks: Number of clicks in the current time step, so number of observation to draw from the Bernoulli

        :return: Realization of the pulled arm, either 0(not buy) or 1(buy)
        :rtype: numpy.ndarray
        """

        phase = self.get_phase()
        self.t += 1
        return super().round_pricing(phase, pulled_arm, n_clicks)

    def get_phase(self):
        """
        Returns the phase of the environment

        :return: Phase of the environment
        :rtype: str
        """

        current_phase = np.searchsorted(np.cumsum(self.phases_duration), self.t % np.sum(self.phases_duration), side='right')
        return 'C' + str(current_phase + 1)
