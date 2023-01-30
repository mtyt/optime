"""Optimization"""
from functools import cached_property, partial
from inspect import signature
import pandas as pd
import numpy as np
from typing import (
    Optional,
    Callable,
    List,
    Union,
    TypeVar,
    Type,
    Any,
    Dict,
    TypedDict,
    NotRequired,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


rng = np.random.default_rng()

# Typing:

Numeric = Union[int, float, np.number]


class GoalDict(TypedDict):
    direction: str
    target: NotRequired[Numeric]


T = TypeVar(
    "T", bound=Any
)  # Anything, but both arguments must be same and function returns the same

# End typing


def child(parent_1: T, parent_2: T) -> T:
    """Produce a child of two parents, based on DNA exchange.
    Each parent must have an attribute called 'dna'. Usually this would just be a
    reference to another attribute, and it should be an array (or list). This function
    will then construct a new instance of the same class with a new dna vector, based on
    a random combination of the parents' dna. But in order to do that, it has to keep
    the attributes that are not the dna the same as the parents. But this can get tricky
    if the parents have different values for some attributes.
    I attempted to solve this by allowing the use of an additional attribute called
    'parent_props' on the parents. If it exists, it should be a list of strings, which
    represents the attribute names of the parents. This function will then obtain these
    attributes from parent_1 and use them in the __init__ of the child.
    If `parent_props` does not exist, the function will look at the signature of the
    __init__ method and check all of the arguments in the parents, but raise an exception
    if any of the values of the attributes are different in the parents.
    If for example the parents have different `name` attributes, this will result in an
    error.

    Args:
        parent_1: An instance of a class, should have a 'dna' attribute
        parent_2: An instance of a class, should have a 'dna' attribute

    Returns:
        A new instance of the same class, with mixed dna.
    """
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
    need to be optimized and as values another dict with 'direction': 'min' or 'max'
    and optionally 'target': a numerical value, which can be used as stop criterion.
    conditions is a list of strings that represent names of columns
    for which the value must be True, otherwise the individual is
    removed from the pareto-front immediately.
    """

    def __init__(
        self,
        individuals: Union[int, List[Any]],
        goals_dict: Dict[str, GoalDict],
        conditions: Optional[List[str]] = None,
        ind_class: Optional[Type[object]] = None,
        possible_dna_values: Optional[List[List]] = None,
    ) -> None:
        """Population is a set of individuals that can be optimized using a Genetic
        Algorithm.

        Args:
            individuals: either a list of instances of a class, or an int.
                In case of int, this many individuals of the ind_class class will be created.
            goals_dict: of the format {"some_performance": {"direction": "min", "target": 0}}
                The optimization goals, which must exist as attributes of the individuals.
                direction can be min/max. target is optional and
                if present, is used for stop criterium and plotting.
            conditions: names of the attributes in the individuals that must be true,
                otherwise individual will be removed from population
            ind_class: if individuals is an int, this class will be used to make
                a number of individuals.
            possible_dna_values: if specified, this lists all possible values for each
                gene in the dna
        """

        if isinstance(individuals, int):
            if ind_class is None:
                raise ValueError(
                    "ind_class must be specified if individuals" "is an int."
                )
            individuals = [ind_class() for i in np.arange(individuals)]
        self._individuals = individuals
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

    @property
    def ind_class(self) -> Type:
        """Returns the class of the individuals."""
        return type(self.indviduals[0])

    @cached_property
    def possible_dna_values(self) -> List[List]:
        """If the possibel DNA values have not been specified at init, derive them
        from all the values in the population."""
        if not self.possible_dna_fixed:
            dna_length = len(self.individuals[0].dna)
            dna_vals = []
            for i in range(dna_length):
                dna_vals.append(list(set([ind.dna[i] for ind in self.individuals])))
            return dna_vals
        else:
            return self.possible_dna_values

    @property
    def goals_names(self) -> List[str]:
        """Returns the keys of the goals_dict."""
        return list(self.goals_dict.keys())

    @property
    def goals_directions(self) -> List:
        """Returns the directions of the goals_dict."""
        return [val["direction"] for _, val in self.goals_dict.items()]

    @property
    def individuals(self) -> List:
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
    def df(self) -> pd.DataFrame:
        """A DataFrame with each row representing an individual, with
        columns all the goals and conditions.
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

    def summary(self, measure: str = "mean") -> dict:
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
            if measure == "mean":
                summary_dict[goal] = np.mean(self.df[goal])
            elif measure == "best":
                direction = self.goals_dict[goal]["direction"]
                if direction == "min":
                    summary_dict[goal] = np.min(self.df[goal])
                elif direction == "max":
                    summary_dict[goal] = np.max(self.df[goal])
                else:
                    ValueError(f"goals should be min or max but got {direction}.")
            else:
                raise ValueError(f"measure should be mean or best but got {measure}.")
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
        self,
        mutprob: float = 0.01,
        mutfunc: Optional[Union[Callable, List[Callable]]] = None,
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

        dna_len = len(self.individuals[0].dna)

        if mutfunc is None:
            # create a RNG for every gene based on a uniform distribution between the
            # min and max values currently in the population (note that this may cause
            # unwanted convergence)
            maxes = [
                max([ind.dna[i] for ind in self.individuals]) for i in range(dna_len)
            ]
            mins = [
                min([ind.dna[i] for ind in self.individuals]) for i in range(dna_len)
            ]

            def random_min_max(a, b):
                """Returns a uniform random number between a and b."""
                return (b - a) * rng.random() + a

            # create a list of functions without arguments. The interval for the random
            # function is fixed by using partial.
            mutfunc = [partial(random_min_max, a, b) for a, b in zip(mins, maxes)]

        if isinstance(mutfunc, list):
            if not all([hasattr(func, "__call__") for func in mutfunc]):
                raise ValueError("mutfunc must be a function or a list of functions")

        elif hasattr(mutfunc, "__call__"):
            mutfunc = [mutfunc for _ in range(dna_len)]
        else:
            raise TypeError("mutfunc should be a function or a list of functions.")

        for ind in self.individuals:
            for index, original_val in enumerate(ind.dna):
                # now check the probability:
                if rng.random() <= mutprob:
                    ind.dna[index] = mutfunc[index]()

    def trim(self, n: Optional[int] = None) -> None:
        """Trim the population down to n individuals, based on Pareto front.

        Args:
            n: resulting number of individuals after trimming
        """
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

    def pareto(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
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

        Args:
            df: if not defined, takes self.df. The DataFrame for which to obtain the
                Pareto Front.

        Returns:
            The Pareto Front of the population.
        """
        if df is None:
            df = self.df

        # remove all Individuals for which at least one of the conditions is
        # not met:
        df = df[df[self.conditions].all(1)]

        crit_cols = self.goals_dict
        front = pd.DataFrame(columns=df.columns)

        for item in df.index:
            union = np.array([])
            # temporarily remove all individuals with the exact same performance as the
            # current item:
            cc = list(crit_cols.keys())
            equal_index = df[(df[cc] == df.loc[item, cc]).all(axis=1)].index
            df_temp = df.drop(index=equal_index)
            for col in crit_cols:
                # Add the individuals that have a (strictly) worse performance on 1
                # criterium to the Union.
                if crit_cols[col]["direction"] == "min":
                    worse_index = df_temp[df_temp[col] > df.at[item, col]].index
                elif crit_cols[col]["direction"] == "max":
                    worse_index = df_temp[df_temp[col] < df.at[item, col]].index
                else:
                    raise ValueError("Values in crit_cols must be 'min' or 'max'")
                union = np.union1d(union, worse_index)
            # the current individual is non-dominated if the index of the df_temp (that
            # is without the individuals with the exact same performance as the current
            # item) equals the index of the union. That means that ALL OTHER individuals
            # are worse on at least 1 criterium.
            nondom = df_temp.index.equals(pd.Index(union))
            if nondom:
                front = pd.concat([front, df.loc[[item], :]], axis=0)

            # remove duplicates from front
            front = front.drop_duplicates(subset=list(self.goals_dict.keys()))
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

    def targets_met(self):
        """Returns whether or not any individual in the population has met all the
        targets. But only if every goal has a target set.
        """
        if not all(["target" in val for _, val in self.goals_dict.items()]):
            return False
        for _, row in self.df.iterrows():
            targets_met = []
            for goal_name, val in self.goals_dict.items():
                if val["direction"] == "min":
                    targets_met.append(row[goal_name] <= val["target"])
                elif val["direction"] == "max":
                    targets_met.append(row[goal_name] >= val["target"])
                else:
                    ValueError(f"Not min or max but {val['direction']}")
            if all(targets_met):
                return True
        return False

    def run(
        self,
        n_gen: int = 10,
        mateprob: float = 1.0,
        mutprob: float = 0.0,
        mutfunc: Optional[Union[Callable, List[Callable]]] = None,
        stop_on_steady_n: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Run the optimization for n_gen generations.

        Args:
            n_gen: The number generation to run the GA for.
            mateprob: Probability for mating. 1 corresponds to 2 parent vectors of
                length equal to population size.
            mutprob: Probability of mutating, should be <=1.
            mutfunc: If necessary, can pass a function or list of functions to generate
                mutation values.
            stop_on_steady_n: if the mean and best values for all criteria don't change
                for this many generations, exit.
            verbose: Turn on or off print statements.

        """
        if stop_on_steady_n is None:
            stop_on_steady_n = n_gen
        summaries: dict = {"mean": [], "best": []}
        for gen in np.arange(n_gen):
            if verbose:
                print(f"Doing generation {gen}.")
            self.make_offspring(mateprob)
            self.mutate(mutprob, mutfunc=mutfunc)
            self.trim()
            summaries["mean"].append(self.summary(measure="mean"))
            summaries["best"].append(self.summary(measure="best"))
            stop_mean = False
            stop_best = False
            # if both the mean and best values haven't changed for 3 generations, stop.
            if gen > stop_on_steady_n - 1:
                mean_stops = []
                best_stops = []
                for goal in self.goals_dict:
                    y_mean = [
                        summ[goal] for summ in summaries["mean"][-stop_on_steady_n::]
                    ]
                    mean_stop = np.array(y_mean).std() < 1e-9
                    mean_stops.append(mean_stop)
                    y_best = [
                        summ[goal] for summ in summaries["best"][-stop_on_steady_n::]
                    ]
                    best_stop = np.array(y_best).std() < 1e-9
                    best_stops.append(best_stop)
                stop_mean = all(mean_stops)
                stop_best = all(best_stops)

            # if all targets are met, stop.
            # for each individual, check if all targets are met.
            stop_targets = self.targets_met()
            stop = (stop_mean and stop_best) or stop_targets
            if stop:
                print("Stop criteria met, stopping early.")
                if stop_mean and stop_best:
                    print(
                        "Mean and best values for goals haven't changed for "
                        f"{stop_on_steady_n} generations"
                    )
                if stop_targets:
                    print(
                        "At least one individual in the population has met all targets."
                    )
                break
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
        for i, goal_name in enumerate(self.summary()):
            y_mean = [gen[goal_name] for gen in self.summaries["mean"]]
            y_best = [gen[goal_name] for gen in self.summaries["best"]]
            y_targ = None
            if goal_name in self.goals_dict:
                if "target" in self.goals_dict[goal_name]:
                    y_targ = [self.goals_dict[goal_name]["target"] for _ in y_mean]
            ax[i].plot(y_mean, "r--", label="mean")
            ax[i].plot(y_best, "r-", label="best")
            if y_targ:
                ax[i].plot(y_targ, "b-", label="target")
            ax[i].set_ylabel(goal_name, rotation=0, ha="right", x=-1)
            ax[i].grid()
            if i == 0:
                ax[i].legend()
            if i == nrows - 1:
                ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
                ax[i].set_xlabel("Generation")
