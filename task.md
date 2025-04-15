PROJECT CONTEXT AND REQUIREMENTS (with AI Agent Focus):
--------------------------------------------------------
We are building an application that allows users to perform financial analysis by uploading CSV files.
Each financial formula is implemented as an independent Python function. Instead of passing a DataFrame directly,
each function receives:
  - ctx: a runtime context object of type RunContext (imported from pydantic_ai).
  - df_name: a string containing the CSV file name.
  
The CSV file is loaded using the following approach:
    from pathlib import Path
    file_path = Path(ctx.deps.data_dir) / df_name

Each function also accepts additional formula-specific parameters (such as column names or optional filter conditions).
The very last parameter in each function is always ctx.

Each function must return a dictionary with three keys:
  - "operation": a string identifier for the formula (e.g., "sum", "average", "vlookup", etc.).
  - "results": a dictionary containing the computed value(s), descriptive message(s), or updated DataFrame details.
  - "formula": a string representing the equivalent Excel formula, to assist users who are familiar with Excel.

Example Excel Equivalents:
    SUM:      =SUM(A1:A10)
    AVERAGE:  =AVERAGE(B1:B10)
    VLOOKUP:  =VLOOKUP("lookup_value", [table], MATCH("return_column", [header], 0), FALSE)

ADDITIONAL INSTRUCTIONS FOR THE AI AGENT:
------------------------------------------
The functions will be invoked by an AI agent, so ensure that each function includes **detailed docstrings**.
Each docstring must explain:
  - The purpose of the function.
  - The role of each parameter (df_name, ctx, and any formula-specific arguments).
  - What the function returns and the structure of the returned dictionary.
  - How errors are handled.
  - The equivalent Excel formula.

In summary, for each financial function:
  1. Load CSV data using df_name and ctx.
  2. Validate required columns.
  3. Optionally filter the data.
  4. Compute the value(s).
  5. Return a dictionary with "operation", "results", and "formula".

Remember to follow the python rules set.