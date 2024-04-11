class DeadOil():

    def __init__(self, oil_api, bubblepoint):
        # define an oil stream that has no live gas in it
        # what it wad before it died

        self.oil_api = oil_api
        self.pbp = bubblepoint

    # does something when you print your class
    def __repr__(self):
        return f'Dead Oil: {self.oil_api} API and {self.pbp} PSIG BubblePoint'

# note, the problem with inheritance is that it inherits your method names
# so if BlackOil inherits properties from FormGas, when I call density, it
# will call the FormGas Density Properties instead of...what I want?
