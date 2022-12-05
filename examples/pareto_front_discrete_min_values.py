import numpy as np
import sys
from os import path
sys.path.insert(0, path.join(path.dirname(__file__), '..'))
print(sys.path)
from optime.ga import child, Population

rng = np.random.default_rng()


class Individual:
    """Dummy class for individuals in the GA."""

    def __init__(self, country=None, language=None, some_property=None):
        self.country = country
        self.language = language
        if some_property is None:
            some_property = [0, 1, 2]
        self.some_property = some_property

    @property
    def some_performance(self):
        """Returns the abs() of difference of the sum of some_property elements
        with 2022."""
        return abs(sum(self.some_property) - 2022)
    
    @property
    def first_gene(self):
        """returns the first gene"""
        return self.some_property[0]
    
    @property
    def second_gene(self):
        """returns the second gene"""
        return self.some_property[1]
    
    @property
    def third_gene(self):
        """returns the third gene"""
        return self.some_property[2]

    @property
    def some_condition(self):
        """We can't allow a value of 1302 in the some_property vector."""
        check = 1302 not in self.some_property
        return check
    
    
dna_length = 3
num_individuals = 10
# prepare num_individuals individuals:
inds = []
# with the current implementation of pareto, all these will be on the front:
inds.append(Individual(some_property=[1, 10, 10]))
inds.append(Individual(some_property=[1, 11, 11]))
inds.append(Individual(some_property=[1, 12, 12]))
inds.append(Individual(some_property=[1, 12, 9]))
inds.append(Individual(some_property=[2, 12, 9]))
inds.append(Individual(some_property=[3, 12, 9]))
inds.append(Individual(some_property=[1, 12, 9]))
# even though the first is clearly better.

# bind the 'dna' attribute to an existing attribute of the Individual class
def get_dna(self):
    return self.some_property

def set_dna(self, val):
    setattr(self, "some_property", val)

Individual.dna = property(fget=get_dna, fset=set_dna)

# set up trivial goals: each gene should be minimized with ultimate target 0.
goals = {"first_gene": {"direction": "min", "target": 0},
         "second_gene": {"direction": "min", "target": 0},
         "third_gene": {"direction": "min", "target": 0}}
conditions = ["some_condition"]

pop = Population(inds, goals, conditions)

print('pop')
print(pop.df)
print('pareto')
pareto = pop.pareto()
print(pareto)


pop.run(30, mutprob=0.2, mutvalues=list(range(0, 13)), verbose=True)

pareto = pop.pareto()
print("pareto after opt")
print(pareto)