# What are recommendations worth? Rebuilding Netflix's study on MovieLens

An independent, educational rebuild of Netflix's [Measuring the Impact of Personalized Recommendations](https://netflixtechblog.com/measuring-the-impact-of-personalized-recommendations-4c26be3a4d96) (paper: [arXiv 2511.07280](https://arxiv.org/abs/2511.07280)), on public data. It adds two things the post doesn't have: uncertainty bands, and a stress test of the key assumption.

## The question

How much viewing do recommendations *cause*? Comparing "recommended vs not recommended" doesn't answer it, because the recommender targets people who'd watch anyway. Netflix's approach:
1. A daily **choice model**: each member picks one title or nothing. Utility = low-rank taste + a **bonus** when a title is recommended.
2. **Identification**: the recommender explores at random, so given what it knew, *which* titles got shown is as good as random. This is an assumption.
3. **Validation** with a nudge A/B test: does the model predict where lost viewing goes?
4. **Counterfactuals**: swap in random, popularity or matrix-factorization recommenders.
5. **Decomposition** into selection, exposure and targeting.

## What's in the notebook

`rec_impact_replication.ipynb` is self-contained. It downloads MovieLens ratings from HuggingFace and runs on CPU in about 5 minutes. Real ratings give taste, popularity, genres and each member's history. Then 40 days of recommendations and viewing are simulated for 4,000 members, with numbered assumptions. Because it's a simulation, the **true** answer is known, so the method can be graded.

## Results (verified run)

| | estimate |
|---|---|
| naive "recommended vs not" multiplier | **7.5×** |
| choice model | **3.4×** |
| truth | **3.1×** |
| choice model, assumption broken | **7.0×** |

- **The dashboard number is 2.4× too high.** The choice model fixes most of it.
- **The leftover gap (+10%) comes from the taste model, not the estimator.** Given true tastes, the same likelihood recovers the bonus.
- **Validation:** predicted vs experimental diversion ratios correlate at 0.77 (Netflix: 0.86). Against the noise-free truth they correlate at 0.99. The gap is experiment noise.
- **Counterfactuals:** the model gets the ranking right, with random worst and matrix factorization best. But **all three 90% bootstrap intervals miss the truth**, and popularity flips sign. Bootstrap covers sampling noise, not model bias.
- **Break the identification assumption** with one hidden signal the recommender uses, and the model gives the naive answer (7.0×) with more confidence.

## What's real vs simulated

**Real:** MovieLens ratings, the taste vectors, popularity, genres, member histories.

**Simulated:** recommendations and viewing (Assumptions 1–5 in the notebook). The model is simplified. It has one canvas instead of several, mean-pooled history instead of an attention model, and a no-rewatch rule. These numbers are not Netflix's numbers. The value is checking the method against a known truth.

## Run it

Open the notebook on Kaggle, Colab or locally, turn internet on and run all.

## Credit

The method belongs to Zielnicki, Aridor, Bibaut, Tran, Chou and Kallus (Netflix). This repo is an unaffiliated educational replication.
