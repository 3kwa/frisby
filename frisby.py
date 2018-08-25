"""
>>> def next_char_from_number_string(s):
...     return chr(int(s.strip()) + 1)
>>> next_char_from_number_string('   64   ')
'A'

>>> def money_to_float(s):
...     return float(s.replace('$', ''))
>>> money_to_float("$30")
30.0
>>> def percent_to_float(s):
...     return float(s.replace('%', '')) * 0.01
>>> percent_to_float("20%")
0.2
>>> def apply_discount(price, discount):
...     return money_to_float(price) * (1 -  percent_to_float(discount))
>>> apply_discount("$30", "20%")
24.0
"""

import abc
import json
import functools

class Functor(metaclass=abc.ABCMeta):
    """
    something that can be mapped over
    """

    @abc.abstractmethod
    def map(self, f):
        """
        identity law :
            F.map(lambda x: x) == F

        composition law:
            F.map(g).map(f) == F.map(lambda x: f(g(x)))
        """

class Monadic(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def of(cls, x):
        pass

    @abc.abstractmethod
    def chain(self, f):
        # flat_map, bind, >>=
        pass

class Box(Monadic):
    """
    >>> Box.of('boxed')
    Box(boxed)
    >>> Box(1).map(lambda x: x)
    Box(1)
    >>> Box(1).fold(lambda x: x + 1)
    2
    >>> def next_char_from_number_string(s):
    ...     return (Box(s)
    ...             .map(lambda s: s.strip())
    ...             .map(int)
    ...             .map(lambda x: int(x) + 1)
    ...             .fold(chr))
    >>> next_char_from_number_string('   64   ')
    'A'
    """

    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x})"

    def __eq__(self, other):
        return self.x == other.x

    @classmethod
    def of(cls, x):
        return cls(x)

    def map(self, f):
        return self.__class__(f(self.x))

    def fold(self, f):
        return f(self.x)

    def chain(self, f):
        return f(self.x)

    def apply(self, other):
        return other.map(self.x)

def apply_discount(price, discount):
    """
    >>> apply_discount("$30", "20%")
    24.0
    """
    return (money_to_float(price)
            .fold(lambda cost:
                (percent_to_float(discount)
                 .fold(lambda savings: cost * (1 - savings)))))

def money_to_float(price):
    return (Box(price)
            .map(lambda s: s.replace('$', ''))
            .map(float))

def percent_to_float(discount):
    return (Box(discount)
            .map(lambda s: s.replace('%', ''))
            .map(lambda x: float(x) * 0.01))

class Either:
    """
    >>> (Right(2)
    ...  .map(lambda x: x + 1)
    ...  .map(lambda x: x / 2)
    ...  .fold(lambda x: "error", lambda x: x))
    1.5
    >>> Left(2).map(lambda x: x + 1).fold(lambda x: "error", lambda x: x)
    'error'

    >>> def find_color(name):
    ...     return from_nullable({"red": "#ff4444",
    ...                           "blue": "#3b5998",
    ...                           "yellow": "fff68f"}.get(name))
    >>> (find_color("blue")
    ...  .map(lambda s: s[1:])
    ...  .fold(lambda x: "no color",
    ...        lambda s: s.upper()))
    '3B5998'
    >>> (find_color("grey")
    ...  .map(lambda s: s[1:])
    ...  .fold(lambda x: "no color",
    ...        lambda s: s.upper()))
    'no color'
    """

    @classmethod
    def of(cls, x):
        return Right.of(x)

class Right:

    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x})"

    def chain(self, f):
        return f(self.x)

    def map(self, f):
        """
        >>> Right(2).map(lambda x: x + 1)
        Right(3)
        """
        return self.__class__(f(self.x))

    def fold(self, f, g):
        return g(self.x)

class Left:

    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x})"

    def chain(self, f):
        return self.__class__(self.x)

    def map(self, f):
        """
        >>> Left(2).map(lambda x: x + 1)
        Left(2)
        """
        return self.__class__(self.x)

    def fold(self, f, g):
        return f(self.x)

def from_nullable(x):
    return Left(x) if x is None else Right(x)

def try_except(f):
    try:
        return Right(f())
    except Exception as e:
        return Left(e)

def read(filename):
    """
    >>> read("config.json")
    '{"port": 8888}\\n'
    """
    with open(filename) as f:
        return f.read()

def get(filename):
    """
    >>> get("config.json")
    8888
    >>> get("no_config.json")
    3000
    >>> get("empty.json")
    3000
    """
    return (try_except(lambda : read(filename))
            .chain(lambda c: try_except(lambda: json.loads(c)))
            .fold(lambda e: 3000,
                  lambda c: c['port']))

# semi-group type with an associative concat method

class SemiGroup(metaclass=abc.ABCMeta):
    """
    A type with an associative concat method
    """

    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x})"

    @abc.abstractmethod
    def concat(self, other):
        pass

class Monoid(SemiGroup, metaclass=abc.ABCMeta):
    """
    A semi group with an empty method (neutral element for concat)
    """

    @abc.abstractmethod
    def empty(cls):
        pass

class Sum(Monoid):
    """
    >>> Sum(1).concat(Sum(2))
    Sum(3)
    >>> Sum.empty().concat(Sum(5))
    Sum(5)
    """

    def concat(self, other):
        return self.__class__(self.x + other.x)

    @classmethod
    def empty(cls):
        return cls(0)

class All(Monoid):
    """
    >>> All(True).concat(All(False))
    All(False)
    >>> All(True).concat(All(True))
    All(True)
    >>> All.empty().concat(All(True))
    All(True)
    >>> All.empty().concat(All(False))
    All(False)
    """

    def concat(self, other):
        return self.__class__(self.x and other.x)

    @classmethod
    def empty(cls):
        return cls(True)


class First(SemiGroup):
    """
    >>> First("first").concat(First("second"))
    First(first)
    """

    def concat(self, other):
        return self.__class__(self.x)


class Map(SemiGroup):
    """
    >>> (Map({"name": First("Curry"), "like": All(False)})
    ...  .concat(Map({"name": "Curry", "like": All(True)})))
    Map({'name': First(Curry), 'like': All(False)})
    """

    def concat(self, other):
        man = {}
        for k, v in self.x.items():
            o = other.x.get(k)
            if o is  None:
                continue
            else:
                man[k] = v.concat(o)
        return self.__class__(man)

class List:
    """
    >>> List.of(Sum(1), Sum(2), Sum(3))
    List([Sum(1), Sum(2), Sum(3)])
    >>> (List.of(Sum(1), Sum(2), Sum(3))
    ...  .fold(Sum.empty()))
    Sum(6)
    >>> List.of(1, 2, 3).map(Sum)
    List([Sum(1), Sum(2), Sum(3)])
    >>> List.of(1, 2, 3).map(Sum).fold(Sum.empty())
    Sum(6)
    >>> List.of(1, 2, 3).fold_map(Sum, Sum.empty())
    Sum(6)
    """

    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x})"

    def __eq__(self, other):
        return all(v == other.x[i] for i, v in enumerate(self.x))

    @classmethod
    def of(cls, *args):
        return cls(list(args))

    def fold(self, empty):
        return functools.reduce(lambda a, b: a.concat(b), self.x, empty)

    def map(self, f):
        return self.__class__([f(x) for x in self.x])

    def fold_map(self, f, empty):
        return functools.reduce(lambda a, b: a.concat(b),
                                (f(x) for x in self.x),
                                empty)

    def apply(self, other):
        return List([y for f in self.x for y in other.map(f).x])


class LazyBox:
    """
    >>> LazyBox(lambda : 'box')
    LazyBox(<lambda>)
    >>> (LazyBox(lambda : 'box')
    ...  .map(lambda x: print('evaluated')))
    LazyBox(<lambda>)
    >>> (LazyBox(lambda : 'boxed')
    ...  .map(lambda x: x.upper())
    ...  .fold(lambda x: x))
    'BOXED'
    """

    def __init__(self, g):
        self.g = g

    def __repr__(self):
        return f"{self.__class__.__name__}({self.g.__name__})"

    def map(self, f):
        return self.__class__(lambda : f(self.g()))

    def fold(self, f):
        return f(self.g())

class Task:
    """
    >>> Task.of(1)
    Task(1)
    >>> (Task.of(1)
    ...  .fork(lambda e: print("error", e),
    ...        lambda x: print("success", x)))
    success 1
    >>> (Task.rejected(1)
    ...  .fork(lambda e: print("error", e),
    ...        lambda x: print("success", x)))
    error 1
    >>> (Task.of(1)
    ...  .map(lambda x: x + 1)
    ...  .fork(lambda e: print("error", e),
    ...        lambda x: print("success", x)))
    success 2
    >>> (Task.rejected(1)
    ...  .map(lambda x: x + 1)
    ...  .fork(lambda e: print("error", e),
    ...        lambda x: print("success", x)))
    error 1
    >>> (Task.of(1)
    ...  .map(lambda x: x + 1)
    ...  .chain(lambda x: Task.of(x + 1))
    ...  .fork(lambda e: print("error", e),
    ...        lambda x: print("success", x)))
    success 3
    >>> (Task.rejected(1) #doctest: +SKIP
    ...  .map(lambda x: x + 1)
    ...  .chain(lambda x: Task.of(x + 1))
    ...  .fork(lambda e: print("error", e),
    ...        lambda x: print("success", x)))
    error 1
    """

    def __init__(self, task):
        # task function with 2 arguments
        # error and success functions with no argument
        self.task = task

    def __repr__(self):
        return f"Task({self.task(lambda e: e, lambda x: x)})"

    @classmethod
    def of(cls, x):
        return cls(lambda error, success: success(x))

    @classmethod
    def rejected(cls, e):
        return cls(lambda error, success: error(e))

    def fork(self, error, success):
        return self.task(error, success)

    def map(self, f):
        def task(error, success):
            return self.task(error, lambda x: success(f(x)))
        return Task(task)

    def chain(self, f):
        return self.task(lambda e: Task.rejected(e),
                         lambda x: f(x))

def launch_missiles():
    """
    >>> (launch_missiles()
    ...  .map(lambda x: x + '!')
    ...  .fork(lambda e: print("error", e),
    ...        lambda x: print("success", x)))
    launch missiles!
    success missile!
    >>> app = launch_missiles().map(lambda x: x + '!') # no output
    >>> app.fork(lambda e: print("error", e),
    ...          lambda x: print("success", x))
    launch missiles!
    success missile!
    >>> (app
    ... .map(lambda x: x + '?')
    ... .fork(lambda e: print("error", e),
    ...       lambda x: print("success", x)))
    launch missiles!
    success missile!?
    """
    def task(error, success):
        print("launch missiles!")
        return success("missile")
    return Task(task)

def read_file(filename):
    """
    >>> app = read_file("config.json")
    >>> app.fork(lambda e: print("error"),
    ...          lambda x: print("success"))
    success
    >>> read_file("no_config.json").fork(lambda e: print("error"),
    ...                                  lambda x: print("success"))
    error
    """
    def task(error, success):
        try:
            with open(filename) as f:
                return success(f.read())
        except Exception as e:
            return error(e)
    return Task(task)

def write_file(filename, content):
    def task(error, success):
        try:
            with open(filename, "w") as f:
                return success(f.write(content))
        except Exception as e:
            return error(e)
    return Task(task)

app = (read_file("config.json")
       .map(lambda x: x.replace('8', '6'))
       .chain(lambda x: write_file("config_6.json", x)))
#app.fork(lambda error: print("error"),
         #lambda success: print("success"))

identity = lambda x : x

class __Functor:
    """
    Functor fx is any type with a map function such as
    fx.map(f).map(g) == fx.map(lambda x: g(f(x)))
    fx.map(identity) == identity(fx)
    prove with Box, Left, Right, List.of, ...

    >>> Box("hipster").map(lambda x: x[-2:]).map(lambda x: x.upper())
    Box(ER)
    >>> Box("hipster").map(lambda x: x[-2:].upper())
    Box(ER)
    >>> Box("hipster").map(identity) == identity(Box("hipster"))
    True
    """

# of is a generic interface to lift a value into a type

def join(monad):
    return monad.chain(identity)

class __Monad:
    """
    >>> monad = Box(Box(Box(3)))
    >>> join(monad.map(join)) == join(join(monad))
    True
    >>> monad = Box('wonder')
    >>> join(Box.of(monad)) == join(monad.map(Box.of))
    True
    """

# currying

def add(x, y):
    return x + y
add = lambda x, y: x + y
inc = lambda y: add(1, y)
add = lambda x: (lambda y: x+ y)
inc =  add(1)

def add(x, y):
    return x + y
inc = functools.partial(add, 1)
assert inc(2) == 3

# Functor = something that can be mapped over
# applicative functor
# F(x).map(f) = F(f).apply(F(x))
assert Box(lambda x: x + 1).apply(Box(2)) == Box(3)
assert Box(lambda x: lambda y: x + y).apply(Box(2)).apply(Box(3)) == Box(5)
assert Box(2).map(lambda x: lambda y: x + y).apply(Box(3)) == Box(5)


assert List.of(lambda x: x + 1).apply(List.of(1, 2, 3)) == List.of(2, 3, 4)
assert (List.of(lambda x: lambda y: lambda z : f"{x}-{y}-{z}")
        .apply(List.of("tee", "shirt"))
        .apply(List.of("large", "medium", "small"))
        .apply(List.of("blue" , "yellow"))) == (
        List([ f"{x}-{y}-{z}" for x in ("tee", "shirt")
                              for y in ("large", "medium", "small")
                              for z in ("blue", "yellow") ]))

# applicative for concurrent actions ... not relevant unless we talked async

# traverse require applicative functor
# re-arrange 2 types

#print(List.of("config.json", "no_config.json")
      #.map(read_file))

"""
List.of(Task(f), Task(f))
List.traverse(Task, f)
Task(List).fork()

f(e) -> t
list.map(f) -> list(t)

for e in list:

"""
