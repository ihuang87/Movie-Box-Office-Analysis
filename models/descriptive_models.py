import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


# import data
movies = pd.read_csv("cleaned_movies.csv")

##  Fitting Models 

# LR (dependent variable is tickets.sold without gross)
# Target
movies["tickets_sold"] = pd.to_numeric(movies["tickets_sold"], errors="coerce")

# Splitting the date up
movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
movies["release_year"]  = movies["release_date"].dt.year
movies["release_month"] = movies["release_date"].dt.month

# Predictors
predictors = ["release_year", "release_month", "Distributor", "Genre", 
              "MPAA","title_has_man", "title_has_love", "title_has_life"]
predictors = [c for c in predictors if c in movies.columns]

# compose X and y
X = movies[predictors].copy()
y = movies["tickets_sold"]

# fit a preliminary model (with all variables available)
formula = (
    "tickets_sold ~ release_year + release_month"
    " +C(Distributor) + C(Genre) + C(MPAA)"
    " + title_has_man + title_has_love + title_has_life"
)

model = smf.ols(formula, data=movies).fit()
print(model.summary())       
pred  = model.predict(movies) 


# Try variable selection using significance of predictors as criteria
alpha = 0.05  # removal threshold
formula_terms = [
    "release_year", "release_month",
    "C(Distributor)", "C(Genre)", "C(MPAA)",
    "title_has_man", "title_has_love", "title_has_life"
]

def fit_formula(terms):
    f = "tickets_sold ~ " + " + ".join(terms)
    return smf.ols(f, data=movies).fit(), f

# start with full model
model, current_formula = fit_formula(formula_terms)

# iteratively drop worst (highest p) term with p > alpha
while True:
    aov = anova_lm(model, typ=2)
    aov = aov.drop(index="Residual")
    # If all p <= alpha, stop
    worst = aov["PR(>F)"].idxmax()
    worst_p = aov.loc[worst, "PR(>F)"]
    if worst_p <= alpha:
        break

    # Remove that whole term from formula
    if worst in formula_terms:
        formula_terms.remove(worst)

    # Refit
    model, current_formula = fit_formula(formula_terms)

print("Final formula:", current_formula)
print(model.summary())
#removed two predictors but retains the same R^2 and other criteria, stick with this one



