# Explanation

1. Prefixes denote what are alternative parts of the same pipeline.
2. Everything that is not an alternative can be joined to create a single DAG.

The rationale is that this way we can introspect this and figure out
what are the different parts of the pipeline, and therefore if we were
to hyperparameter search over the pipeline, we could know what degree
of freedom we have with respect to modules.

# open questions

## How do we handle shared code between the modules? 
e.g. how should one structure the repository?

ideas:

1. have a directive at the top of the file that specifies this is or is not Hamilton code.
2. make people place it in a separate directory with a directive to mark it.
3. a combination of the two.

 