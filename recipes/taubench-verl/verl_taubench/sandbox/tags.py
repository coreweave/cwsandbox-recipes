"""Tags every sandbox this recipe creates carries.

A leaked sandbox must be identifiable without knowing which run made it, so
each one gets the stable RECIPE_TAG plus its run-scoped tag.
"""

RECIPE_TAG = "verl-taubench"
