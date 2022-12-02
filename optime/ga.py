"""Optimization"""
from functools import cached_property
from inspect import signature
import pandas as pd
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


rng = np.random.default_rng()


def sort_pareto(df, crit_cols):
    """Find the Pareto front and assign all the items the order 0,
    then remove these items from the df and find the next PF, whih is
    assigned order 1, etc...
    NOT YET IMPLEMENTED
    """


def child(parent_1, parent_2):
    """Produce a child of two parents, based on DNA exchange"""
    if not isinstance(parent_1, type(parent_2)):
        raise TypeError("To inputs must be the same class!")
    child_cls = parent_1.__class__
    try:
        _ = len(parent_1.dna)
    except AttributeError as err:
        raise AttributeError("Did you define the dna property on the" "class?") from err

    # do some random gene swapping:
    both = np.vstack((parent_1.dna, parent_2.dna))
    both_perm = rng.permuted(both, axis=0)
    child_vec = both_perm[0]

    child_par = {}  # a dict that will contain the parameters

    # Two alternatives to get the __init__ arguments from the parents
    # 1) if a parent_1.parent_props is defined, use that list to
    # obtain them from parent_1, assuming that either parent is fine
    try:
        parent_props = parent_1.parent_props
        for prop in parent_props:
            child_par[prop] = getattr(parent_1, prop)
    except AttributeError:

        # 2) if that doesn't exist, use the method below:
        # try to construct an instance of the child class with all of
        # its __init__ arguments the same as the parents, if they are
        # the same for both parents, otherwise don't set it and hope
        # it's not required.
        sig = signature(child_cls.__init__)
        for par in sig.parameters.keys():
            if not par == "self":
                parent_1_par = getattr(parent_1, par)
                parent_2_par = getattr(parent_2, par)
                test = parent_1_par == parent_2_par
                try:
                    test = test.all()
                except AttributeError:
                    pass
                if test:  # if they're different, they could be the DNA
                    child_par[par] = parent_1_par

    kid = child_cls(**child_par)
    kid.dna = child_vec
    return kid


class Population:
    """goals_dict is a dictionnary with as keys the column names that
    need to be optimized and as values 'min' or 'max'.
    conditions is a list of strings that represent names of columns
    for which the value must be True, otherwise the individual is
    removed from the pareto-front immediately.
    """

    def __init__(
        self,
        individuals=None,
        goals_dict=None,
        conditions=None,
        ind_class=None,
        possible_dna_values=None,
    ):
        if individuals is None:
            individuals = 10
        if isinstance(individuals, int):
            if ind_class is None:
                raise ValueError(
                    "ind_class must be specified if individuals" "is None or an int."
                )
            individuals = [ind_class() for i in np.arange(individuals)]
        self._individuals = individuals  # a list of Recipe objects
        self.original_size = len(individuals)
        self.goals_dict = goals_dict
        if conditions is None:
            conditions = []
        self.conditions = conditions
        self.summaries = None
        if possible_dna_values is not None:
            self.possible_dna_fixed = True
            self.possible_dna_values = possible_dna_values
        else:
            self.possible_dna_fixed = False

    @cached_property
    def possible_dna_values(self):
        """If the possibel DNA values have not been specified at init, derive them
        from all the values in the population."""
        if not self.possible_dna_fixed:
            dna_length = len(self.individuals[0].dna)
            dna_vals = []
            for i in range(dna_length):
                dna_vals.append(list(set([ind.dna[i] for ind in self.individuals])))
            return dna_vals

    @property
    def goals_names(self):
        """Returns the keys of the goals_dict."""
        return list(self.goals_dict.keys())

    @property
    def individuals(self):
        """Returns the _individuals."""
        return self._individuals

    @individuals.setter
    def individuals(self, x):
        self._individuals = x
        list_of_dependent_properties = ["df", "df_flat", "possible_dna_values"]
        for prop in list_of_dependent_properties:
            if prop in self.__dict__:
                delattr(self, prop)

    @cached_property
    def df(self):
        """A DataFrame with each row representing a Recipe, with
        columns of kg of each food, the impact and enough_score.
        """
        df_pop = pd.DataFrame(columns=["Individual"] + self.goals_names)
        for i, ind in enumerate(self.individuals):
            df_pop.loc[i, "Individual"] = ind
            df_pop.loc[i, self.goals_names] = [
                getattr(ind, name) for name in self.goals_names
            ]
            df_pop.loc[i, self.conditions] = [
                getattr(ind, name) for name in self.conditions
            ]
        return df_pop

    def summary(self, measure: str = 'mean') -> dict:
        """Returns a dict which is a summary of the performance of the
        Population as a mean of each parameter.
        
        Args:
            measure: 'mean' or 'best'
            
        Returns:
            dict with the mean or best for each parameter, plus which percentage of
            conditions are met.
        """
        
        summary_dict = dict()
        for goal in self.goals_dict:
            if measure == 'mean':
                summary_dict[goal] = np.mean(self.df[goal])
            elif measure == 'best':
                target = self.goals_dict[goal]
                if target == 'min':
                    summary_dict[goal] = np.min(self.df[goal])
                elif target == 'max':
                    summary_dict[goal] = np.max(self.df[goal])
                else:
                    ValueError(f'goals should be min or max but got {target}.')
            else:
                raise ValueError(f'measure should be mean or best but got {measure}.')
        for cond in self.conditions:
            summary_dict[cond] = sum(self.df[cond]) / len(self.df)
        return summary_dict

    def make_offspring(self, mateprob: float = 1) -> None:
        """Make n children fron 2n random parents.

        Args:
            mateprob: Probability for mating. 1 corresponds to 2 parent vectors of
                length equal to population size.

        """
        if mateprob < 0 or mateprob > 1:
            raise ValueError(
                f"mateprob should be min 0 and max 1 (received {mateprob})."
            )
        n = int(self.original_size * mateprob)
        # Make 2 random vectors of length n with values 0 to
        # len(self.individuals) to determine who mates with whom:
        parent_1_vec = rng.choice(self.individuals, n)
        parent_2_vec = rng.choice(self.individuals, n)
        rec = self.individuals
        for parent_1, parent_2 in zip(parent_1_vec, parent_2_vec):
            rec.append(child(parent_1, parent_2))
        self.individuals = rec

    def mutate(
        self, mutprob: float = 0.01, values: Optional[list | list[list]] = None
    ) -> None:
        """Mutate the DNA of the individuals of the population. Each gene has a
        probability 'prob' of mutating. If values is specified, it lists all possible
        values for each gene if it's a list of lists. If it's just a list, it's assumed
        that all values are valid for all genes. If it is None, we assume it's binary.
        If the number of values is >2, it's the probability that ANOTHER value is
        uniformly picked.

        Args:
            mutprob: Probability of mutating, should be <=1.
            mutvalues: If necessary, can pass possible values for each element in the DNA
        """

        if mutprob < 0 or mutprob > 1:
            raise ValueError(f"mutprob should be min 0 and max 1 (received {mutprob}).")

        if values is None:
            values = [0, 1]
        if isinstance(values, list):
            if not all([isinstance(val, list) for val in values]):
                # Either it's just a list of values:
                if not all([isinstance(val, type(values[0])) for val in values]):
                    raise ValueError(
                        "values is not a list of list nor a list of the same type"
                    )
                else:
                    flat_list = True
            else:
                flat_list = False

        for ind in self.individuals:
            for index, original_val in enumerate(ind.dna):
                if flat_list:
                    gene_values = values.copy()
                else:
                    gene_values = values[index].copy()
                if original_val in gene_values:
                    gene_values.remove(original_val)
                # now check the probability:
                if (rng.random() <= mutprob) and gene_values:
                    ind.dna[index] = rng.choice(gene_values)

    def trim(self, n=None):
        """Trim the population down to n individuals, based on Pareto front"""
        if n is None:
            n = self.original_size
        new_n = 0
        temp_pop = self.df.copy()
        new_pop = pd.DataFrame(columns=self.df.columns)
        while new_n < n:
            front = self.pareto(df=temp_pop)
            # the front is added to the new population:
            new_pop = pd.concat([new_pop, front])
            # check the size of the new population:
            new_n = len(new_pop)
            # remove the front from the temp_pop and continue with that:
            temp_pop = temp_pop.drop(front.index)

        self.individuals = list(new_pop.iloc[0:n]["Individual"].values)
        # print('Indices on df have been reset')

    def pareto(self, df=None):
        """df is the dataframe on which to select the pareto front.
        crit_cols is a dict with the keys being the column names of df
        according which to sort, and the values being 'min' or 'max',
        depending on whether the value should be minimized or maximized.

        The algo works as follows:
        - Go through each item in the DataFrame to determine whether it is
        dominated
        - Non-dominated means that every other point is not better on at least
        1 criterium
        - This is tested by sorting the df by one criterium and select all
        points that are worse, then sort by the next criterium and select all
        the points that are worse and add them to the union. If all criteria
        are done, compare the union to the original collection and if they
        contain the same points, that means all points are worse on at least
        1 criterium and the point is non-dominated.

        TODO: allow list for crit_cols, assuming all are 'min'
        TODO: rethink definition of Pareto, or allow 2. Because now I can have
        individuals with the same score on 1 criterium (0) but that are clearly
        better than others because of another criterium, but they're all part
        of the front.
        """
        if df is None:
            df = self.df

        # remove all Individuals for which at least one of the conditions is
        # not met:
        df = df[df[self.conditions].all(1)]

        crit_cols = self.goals_dict
        front = pd.DataFrame(columns=df.columns)
        worse_dict = {}  # dict containing a sorted df per criterium
        for col in crit_cols:
            if crit_cols[col] == "min":
                worse_dict[col] = df.sort_values(col, ascending=True)
            elif crit_cols[col] == "max":
                worse_dict[col] = df.sort_values(col, ascending=False)
            else:
                raise ValueError("Values in crit_cols must be 'min' or 'max'")

        for item in df.index:
            union = np.array([])
            for col in crit_cols:
                worse_index = worse_dict[col].loc[item::].index
                # remove all rows that have the same value on col as item:
                val = worse_dict[col].loc[item, col]
                same_val = worse_dict[col][worse_dict[col][col] == val]
                same_val_index = same_val.index
                # worse actually means not strictly better
                worse_index = np.union1d(worse_index, same_val_index)
                union = np.union1d(union, worse_index)
            nondom = df.index.equals(pd.Index(union))
            if nondom:
                front = pd.concat([front, df.loc[[item], :]], axis=0)
        return front

    def plot_pareto(self, mode="2d"):
        """Plot the pareto front on top of the complete population. In case
        of 2 variables, 2D plot is the only valid option. For more variables,
        a 3D plot can be made, or multiple 2D plots.
        """
        df = self.df
        front = self.pareto()[self.goals_dict.keys()]
        num_vars = len(front.columns)
        if num_vars == 2:
            fig, ax = plt.subplots()
            x = front.columns[0]
            y = front.columns[1]
            df.plot.scatter(x=x, y=y, ax=ax, label="All Data")
            front.plot.scatter(x=x, y=y, color="r", ax=ax, label="Pareto")
            ax.grid()
            ax.legend()
        elif num_vars > 2:
            if num_vars > 3:
                # force mode to 2d
                mode = "2d"

            if mode == "2d":
                fig, ax = plt.subplots(nrows=num_vars, ncols=1)
                for x_i, y_i in [(i, np.mod(i + 1, num_vars)) for i in range(num_vars)]:
                    x = front.columns[x_i]
                    y = front.columns[y_i]
                    df.plot.scatter(x=x, y=y, ax=ax[x_i], label="All Data")
                    front.plot.scatter(x=x, y=y, color="r", ax=ax[x_i], label="Pareto")
                    ax[x_i].grid()
                    ax[x_i].legend()
            elif mode == "3d":
                x = front.columns[0]
                y = front.columns[1]
                z = front.columns[2]
                fig = plt.figure()
                ax = fig.add_subplot(projection="3d")
                ax.scatter(df[x], df[y], df[z])
                ax.scatter(front[x], front[y], front[z], color="red")
        else:
            raise ValueError("Need at least 2 variables to plot.")

        plt.show()

    def run(
        self,
        n_gen: int = 10,
        mateprob: float = 1.0,
        mutprob: float = 0.01,
        mutvalues: Optional[list | list[list]] = None,
        verbose: bool = False,
    ) -> None:
        """Run the optimization for n_gen generations.

        Args:
            n_gen: The number generation to run the GA for.
            mateprob: Probability for mating. 1 corresponds to 2 parent vectors of
                length equal to population size.
            mutprob: Probability of mutating, should be <=1.
            mutvalues: If necessary, can pass possible values for each element in the DNA
            verbose: Turn on or off print statements.

        """
        summaries = {'mean':[], 'best':[]}
        for gen in np.arange(n_gen):
            if verbose:
                print(f"Doing generation {gen}.")
            self.make_offspring(mateprob)
            self.trim()
            self.mutate(mutprob, mutvalues)
            summaries['mean'].append(self.summary(measure='mean'))
            summaries['best'].append(self.summary(measure='best'))
        self.summaries = summaries

    def plot_progress(self, fig=None, ax=None):
        """Plot the progress of the generations."""
        nrows = len(self.summary())
        if fig is None and ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=nrows, sharex=True)
        else:
            if not len(ax) == nrows:
                raise ValueError(
                    f"ax has length {len(ax)} but summary has length {nrows}"
                )

        fig.set_size_inches(8, 8)
        for i, summ in enumerate(self.summary()):
            y_mean = [gen[summ] for gen in self.summaries['mean']]
            y_best = [gen[summ] for gen in self.summaries['best']]
            
            ax[i].plot(y_mean, label='mean')
            ax[i].plot(y_best, label='best')
            ax[i].set_ylabel(summ, rotation=0, ha='right', x=-1)
            ax[i].grid()
            if i == 0:
                ax[i].legend()
            if i == nrows-1:
                ax[i].set_xticks(list(range(len(y_mean))))
                ax[i].set_xticklabels(list(range(len(y_mean))))
