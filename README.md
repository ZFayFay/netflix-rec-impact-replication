# Testing Netflix's recommendation-impact method in a simulation with a known answer

An independent rebuild of the method in [The Value of Personalized Recommendations: Evidence from Netflix](https://arxiv.org/abs/2511.07280) (Zielnicki, Aridor, Bibaut, Tran, Chou and Kallus), also described on the Netflix Tech Blog in [Measuring the Impact of Personalized Recommendations](https://netflixtechblog.com/measuring-the-impact-of-personalized-recommendations-4c26be3a4d96). Viewing is simulated on top of real MovieLens ratings, so the true effect of a recommendation is known and the method can be graded.

All viewing here is simulated. Nothing in this repository says anything about Netflix's real numbers.

## The question

How much viewing do recommendations cause? Comparing "recommended" with "not recommended" doesn't answer it, because the recommender shows people what they were likely to watch anyway. The paper models the choice itself: each day each member picks one title or nothing, and a title's appeal is the member's taste plus a push when it is recommended. It treats what the recommender shows as random once you account for what the model sees, validates the model against an A/B test that shifts recommendations toward one category of titles, and uses it to compare recommenders.

## Setup

About 890,000 MovieLens ratings give real tastes, popularity, genres and member histories. On top of them the notebook simulates 40 days of recommendations and viewing for 4,000 members, with every assumption numbered. The true push roughly triples the odds of watching, in line with the paper. The simulated members choose the way the model assumes, so this is the friendliest possible test of the method.

## Results

| | Multiplier on viewing |
|---|---|
| Naive "recommended vs not" | 7.5× |
| Choice model | 3.4× |
| Truth | 3.1× |
| Choice model, one signal hidden from it | 7.0× (naive 12.3×, truth 3.1×) |

- In two more random worlds the model said 3.5× and 3.7× against truths of 3.1× and 3.2×. It removes most of the bias and overshoots by 10% to 16%.
- The overshoot comes from rebuilding taste from viewing history, not from the estimator. Given the true tastes and fit to convergence, the estimator lands within about 1% of the true push. With 16,000 members the overshoot falls to 4%.
- The nudge validation needs a big experiment. With 100 simulated days per genre the model matches the experiment at 0.77 (the paper reports 0.86 on real data). With experiments of 5 to 50 days the match ranges from 0.09 to 0.63. A model with no personalization matches the noise-free truth at 0.93 to 0.95, so a high correlation alone doesn't show the model is good.
- As the hidden signal grows, the estimate moves from 3.5× to 3.7×, 4.2× and 7.0× while the truth stays at 3.1×.
- The model orders the alternative recommenders correctly but is too optimistic about popularity and matrix factorization by about one to two points. In this simulated world matrix factorization beats the current recommender, unlike in the paper.

## Files

- `rec_impact_replication.ipynb`: the rebuild, executed. It downloads MovieLens from Hugging Face and runs on a CPU in about five minutes.
- `checks/`: scripts that rerun the notebook's cells in more worlds, with more members, at four hidden-signal strengths and with different experiment lengths, and their saved results. Section 10 of the notebook reads them.
- `figures/`: charts.
- `medium_draft.md`: the blog post.

## Run it

```bash
pip install numpy pandas scipy matplotlib torch pyarrow
jupyter nbconvert --to notebook --execute --inplace rec_impact_replication.ipynb
```

## Credit

The method belongs to Zielnicki, Aridor, Bibaut, Tran, Chou and Kallus at Netflix. This is an independent rebuild in simulation. Any mistakes are mine.
