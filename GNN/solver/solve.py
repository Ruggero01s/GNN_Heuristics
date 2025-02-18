from unified_planning.io import PDDLReader
from unified_planning.shortcuts import *

reader = PDDLReader()
problem = reader.parse_problem('solver/data_pddl/blocks/domain.pddl', 'solver/data_pddl/blocks/p01.pddl')
print(problem)
with OneshotPlanner(problem_kind=problem.kind) as planner:
    result = planner.solve(problem)
    if result.status in unified_planning.engines.results.POSITIVE_OUTCOMES:
        print(f"{planner.name} found this plan: {result.plan}")
    else:
        print("No plan found.")
