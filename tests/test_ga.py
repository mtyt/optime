'''Unit tests for the ga.py module'''
import unittest
import numpy as np
from optime.ga import child, Population

rng = np.random.default_rng()


class Individual():
    '''Dummy class for individuals in the GA.'''
    def __init__(self, country=None, language=None, some_property=None):
        self.country = country
        self.language = language
        if some_property is None:
            some_property = [0,1,2]
        self.some_property = some_property
    
    @property
    def some_performance(self):
        '''Returns the abs() of difference of the sum of some_property elements
        with 2022.'''
        return abs(sum(self.some_property)-2022)
    
    @property
    def some_condition(self):
        '''We can't allow a value of 1302 in the some_property vector.'''
        check = 1302 not in self.some_property
        return check


class TestChild(unittest.TestCase):
    '''Unit tests for Child class.'''
    def test(self):        
        i_1 = Individual('Belgium', 'Dutch')
        i_2 = Individual('Belgium', 'Dutch')
        # give both some dna:
        i_1.dna = [1,1,1,1]
        i_2.dna = [0,0,0,0]
        
        # go through 2 cases: with and without parent_props:
        # Do False first because otherwise the parent_props exist.
        for with_parent_props in [False, True]:
            if with_parent_props:
                # Define that the attributes can just be copied from a parent:
                Individual.parent_props = ['country', 'language']
            kid = child(i_1, i_2)
            self.assertIsInstance(kid, Individual)
            # check if each element of dna occurs in either i_1 or i_2
            for gene in kid.dna:
                check = (gene in i_1.dna) or (gene in i_2.dna)
                self.assertTrue(check)
                
                
class TestPopulation(unittest.TestCase):
    '''Unit tests for Population class.'''
    dna_length = 3
    some_prop = rng.integers(0, 2000, dna_length)
    some_prop
    num_individuals = 10
    # prepare num_individuals individuals:
    inds = []
    for _ in range(num_individuals):
        inds.append(Individual('Belgium', 'Dutch',
                       some_property = some_prop))
    
    # bind the 'dna' attribute to an existing attribute of the Individual class
    def get_dna(self):
        return self.some_property
    def set_dna(self, val):
        setattr(self, 'some_property', val)
    Individual.dna = property(fget=get_dna, fset=set_dna)
    
    goals = {'some_performance':'min'}
    conditions = ['some_condition']
    
    pop = Population(inds, goals, conditions)
    
    def test(self):
        pop = TestPopulation.pop
        self.assertEqual(pop.goals_names, ['some_performance'])
        # check if the DataFrame is constructed correctly:
        for col in ['Individual', 'some_performance', 'some_condition']:
            check = col in pop.df.columns
            self.assertTrue(check)
        self.assertEqual(len(TestPopulation.inds), len(pop.df))
        
        # now just run some of the methods without error:
        _ = pop.summary()
        pop.make_offspring()
        pop.trim(n=12)
        self.assertEqual(len(pop.df), 12)
        pop.mutate(prob = 1, values=[6]) # should turn all DNA values into 6
        self.assertTrue(all([all([val == 6 for val in ind.dna]) for ind in pop.individuals]))
        # now also the DNA values should return list of 6:
        self.assertTrue(pop.possible_dna_values == [[6]]*TestPopulation.dna_length)
        pop.mutate(prob = 0, values=[8000]) # should turn NO DNA values into 8:
        self.assertTrue(all([all([val != 8000 for val in ind.dna]) for ind in pop.individuals]))
        pop.mutate(prob = 1, values=[[8000, 166], [8000], [8000]]) # should turn All DNA values into 8000 or 166:
        self.assertTrue(all([all([val == 8000 or val == 166 for val in ind.dna]) for ind in pop.individuals]))
        
    def test_init(self):
        def cause_err():
            pop_err = Population() # should raise ValueError
        self.assertRaises(ValueError, cause_err)
        pop_0 = Population(ind_class=Individual) # should create pop with 10 individuals
        self.assertEqual(len(pop_0.individuals), 10)
        pop_1 = Population(2, ind_class=Individual) # should create pop with 2 individuals
        self.assertEqual(len(pop_1.individuals), 2)