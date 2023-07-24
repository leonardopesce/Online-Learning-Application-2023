from Environment import *


class NonStationaryEnvironment(Environment):
    # probabilities -> prob matrix that include for each phase the mean of all arms
    # capire come gestire le probabilità perchè qui ho solo una categoria con varie probabilità mentre prima avevo tante categorie con solo una probabilità
    # forse mi basta fare l'override dei metodi che usano probability?
    # alternativa è di cambiare l'environmnet per essere generico per ogni classe e poi fare una sottoclasse nel caso del context
    # oppure la classe context environment puo contenere tanti environment quanti sono i context
    # oppure come faccio adesso metto un dizionario con chiave la fase
    def __init__(self, n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs, phases_duration):
        super().__init__(n_prices, prices, probabilities, bids_to_clicks, bids_to_cum_costs, other_costs)
        self.t = 0
        self.phases_duration = phases_duration  # list where each element is the length of the phase
        self.n_phases = len(self.phases_duration)

    def round_pricing(self, pulled_arm, n_clicks=1):
        phase = self.get_phase()
        self.t += 1
        return super().round_pricing(phase, pulled_arm, n_clicks)

    def get_phase(self):
        current_phase = np.searchsorted(np.cumsum(self.phases_duration), self.t, side='right')
        return 'C' + str(current_phase + 1)
