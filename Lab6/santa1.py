# %%
from nltk import Prover9, Prover9Command
from nltk.sem import Expression

read_expr = Expression.fromstring

p = Prover9()
p._binary_location = r"C:\Program Files (x86)\Prover9-Mace4\bin-win32"


# %%
assumptions = [
    read_expr(p)
    for p in [
        # Every child loves Santa.
        """
        all x.( child(x) -> loves(x,Santa) )
        """,
        # Everyone who loves Santa loves any reindeer.
        """
        all x.(
            loves(x,Santa) -> (
                all y.( reindeer(y) -> loves(x,y))
            )
        )
        """,
        # Rudolph is a reindeer, and Rudolph has a red nose.
        """
        reindeer(Rudolph) & rednose(Rudolph)
        """,
        # Anything which has a red nose is weird or is a clown.
        """
        all x.(rednose(x) -> weird(x) | clown(x))
        """,
        # No reindeer is a clown.
        """
        -exist x.(reindeer(x) & clown(x))
        """,
        # Scrooge does not love anything which is weird.
        """
        all x.( weird(x) -> -loves(Scrooge,x))
        """,
    ]
]

goal = read_expr(
    # Scrooge is not a child.
    """
    -child(Scrooge)
    """
)


# %%
prover = Prover9Command(goal=goal, assumptions=assumptions, prover=p)

prover.print_assumptions()

print()
is_proved = prover.prove()

if is_proved:
    print("It can be proved.\nSee below:\n")
    print(prover.proof())
else:
    print("It cannot be proved")
