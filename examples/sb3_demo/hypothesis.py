"""Does the predicted gamma advantage hold at two training budgets?

The two edges remain separate. Nothing here chooses files, starts
training, or treats the two budgets as independent replications.
"""
from corroborate.core.finding import Finding

from examples.sb3_demo.early_claim import higher_gamma_improves_early_return
from examples.sb3_demo.sb3_claim import higher_gamma_improves_return

BRIDGES = (higher_gamma_improves_early_return, higher_gamma_improves_return)
FINDINGS: tuple[Finding, ...] = ()
