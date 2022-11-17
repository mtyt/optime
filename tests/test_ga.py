'''Unit tests for the ga.py module'''
import unittest
import numpy as np
from optime import child, Population

rng = np.random.default_rng()


class Individual():
    '''Dummy class for individuals in the GA.'''
    def __init__(self, country, language, some_property=None):
        self.country = country
        self.language = language
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
        for with_parent_props in [True, False]:
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
    
    # prepare 10 individuals:
    inds = [Individual('Belgium', 'Dutch',
                       some_property = rng.integers(0,2000, 7))
            for _ in range(10)]
    
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
        self.assertEquals(pop.goals_names, ['some_performance'])
        # check if the DataFrame is constructed correctly:
        for col in ['Individual', 'some_performance', 'some_condition']:
            check = col in pop.df.columns
            self.assertTrue(check)
        self.assertEqual(len(TestPopulation.inds), len(pop.df))
        
        # now just run some of the methods without error:
        _ = pop.summary()
        pop.make_offspring()
        pop.trim()