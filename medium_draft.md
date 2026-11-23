# Your dashboard says recommendations multiply viewing by 7.5×. The truth is 3.1×.

## I rebuilt Netflix's method in a simulation where I know the true answer. Then I tried to break it.

*Part 3 of **Netflix Tech Blog, rebuilt**. [Part 1: artwork personalization](https://medium.com/@faycal.zouine.usa/i-rebuilt-netflixs-artwork-personalization-system-on-movielens-here-s-what-actually-matters-d0540b7c43eb) · [Part 2: the billboard bandit](https://medium.com/@faycal.zouine.usa/i-rebuilt-netflixs-billboard-bandit-the-metric-picked-the-winner-686296a3b72f)*

*6-minute read. In a hurry? Read the bold lines.*

Every recommender team gets asked the same question: what are recommendations worth?

And every team has the same answer ready. A dashboard. Compare viewing when a title was recommended vs when it wasn't. Done.

In my rebuild, that dashboard says recommendations multiply viewing by **7.5×**.

The truth is **3.1×**.

That's not a rounding error. That's a team taking credit for viewing that would have happened anyway (selection bias). The recommender shows you what you already like. Of course the "recommended" group watches more.

> Recommended is not the same as caused.

I've spent 8 years in marketing measurement, where the question is always some version of "what did this cause?" So when Netflix published [how they answer it](https://netflixtechblog.com/measuring-the-impact-of-personalized-recommendations-4c26be3a4d96), I rebuilt it.

## Netflix's answer, in plain words

They model the choice itself. Each day, each member picks one title, or nothing. A title's appeal is your taste for it, plus a **bonus** if it's recommended today. That bonus is what recommendations cause.

The key assumption: the recommender adds a bit of randomness to what it shows. So once you account for what it knew about you, *which* titles got shown is close to a coin flip. That randomness is what separates "shown because you'd like it" from "watched because it was shown."

They check the model with a small A/B test that nudges recommendations away from some titles, and see if the model predicts where that viewing goes. Then they ask: what if we swapped our recommender for random, popularity, or plain matrix factorization? Their answer: −16%, −12%, −4%.

## Why a simulation

I can't see Netflix's data. And in a simulation, I know the truth.

I built the world from 890k real MovieLens ratings: real tastes, real popularity, real genres, real viewing histories. Then I simulated 40 days of recommendations and viewing for 4,000 members, with every assumption written down.

**That turns "does the method work?" into a question I can actually answer.**

## What held up

**The model fixes most of the gap.** Dashboard: 7.5×. Model: 3.4×. Truth: 3.1×.

![Naive vs model](https://raw.githubusercontent.com/ZFayFay/netflix-rec-impact-replication/main/figures/fig1_naive_vs_model.png)

**The A/B check works, if the test is big enough.** I simulated Netflix's nudge test on 8 genres. The model's predictions matched the test at 0.77 correlation (Netflix got 0.86), and matched the noise-free truth at 0.99. The gap is test noise, not the model. About a quarter of the viewing a genre lost went to "watch nothing." The rest moved to other titles.

## What didn't

**1. The leftover 10% hides in the taste model.** I gave the same method the *true* tastes, and it landed close to the true bonus (1.02 vs 1.10). So the bias comes from rebuilding taste out of viewing history. My first, simpler version was off by 24%. A better one, by 11%.

**The assumption can hold perfectly and you still get bias, if your model can't see what the recommender saw.**

**2. The error bars lied.** Netflix reports no uncertainty. So I added the standard thing: refit on 10 resamples of members (bootstrap) and report a 90% range. All three ranges missed the truth.

![Counterfactuals](https://raw.githubusercontent.com/ZFayFay/netflix-rec-impact-replication/main/figures/fig3_counterfactuals.png)

The ranking was right: random is worst, matrix factorization best. But popularity flipped sign (+0.8% predicted, −0.4% true). Bootstrap measures sampling noise. It says nothing about a biased model.

**A tight range around a wrong answer looks rigorous. It's still wrong.**

**3. Break the assumption and you're back to the dashboard.** I added one hidden signal: the recommender knows something about you that the model doesn't. Think of a search you just did. The model's answer jumped from 3.4× to **7.0×**. Almost the dashboard number, now with a fancy model behind it.

## What to remember

1. **Never report "recommended vs not" as impact.** It's selection plus effect.
2. **List every signal the recommender uses.** If your model is missing one, your estimate is too high.
3. **Validate with a nudge test.** It's the only check that touches reality.
4. **A bootstrap range is not a range on the truth.** Say so when you show one.

---

*Notebook and code: [github.com/ZFayFay/netflix-rec-impact-replication](https://github.com/ZFayFay/netflix-rec-impact-replication). The method belongs to the Netflix authors (Zielnicki, Aridor, Bibaut, Tran, Chou, Kallus). This is an independent rebuild of their [post](https://netflixtechblog.com/measuring-the-impact-of-personalized-recommendations-4c26be3a4d96) and [paper](https://arxiv.org/abs/2511.07280).*
