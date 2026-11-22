# Your dashboard says recommendations multiply viewing by 7.5×. The truth is 3.1×.

*Netflix published a method to measure how much viewing recommendations actually cause. I rebuilt it on MovieLens, in a world where I know the true answer. Then I tried to break it. Code on [GitHub](https://github.com/ZFayFay/netflix-rec-impact-replication).*

---

## The question every recommender team gets asked

"How much do recommendations drive?"

The easy answer is a dashboard. Compare the watch rate of titles when they were recommended vs when they weren't. In my rebuild that says recommendations multiply viewing by **7.5×**.

The true number is **3.1×**.

The gap is selection. The recommender shows you titles you were likely to watch anyway. So the "recommended" group was always going to watch more. Anyone who's done marketing attribution has seen this movie. Last-touch gives the credit to the channel that showed up for people already converting.

## Netflix's answer

Netflix ([post](https://netflixtechblog.com/measuring-the-impact-of-personalized-recommendations-4c26be3a4d96), [paper](https://arxiv.org/abs/2511.07280)) models the choice itself. Each day each member picks one title, or nothing. A title's appeal is your taste for it plus a **bonus** if it's recommended today. The bonus is what recommendations cause.

The key assumption: the recommender explores at random. So once you account for what it knew about you, *which* titles got shown is as good as a coin flip. That randomness is what separates "shown because you'd like it" from "watched because it was shown".

They validate it with a small A/B test that nudges recommendations away from some titles. Then they check if the model predicts where the lost viewing goes. Then they ask: what if we swapped our recommender for random, popularity, or plain matrix factorization? Their answer: −16%, −12%, −4%.

## Why rebuild it in a simulation?

Because I can't see Netflix's data. And because in a simulation I know the truth. I built the world from 890k real MovieLens ratings: real tastes, real popularity, real genres, real viewing histories. Then I simulated 40 days of recommendations and viewing for 4,000 members, with every assumption written down.

That turns "does the method work?" into a question I can actually answer.

## What held up

**The model fixes most of the bias.** Naive: 7.5×. Model: 3.4×. Truth: 3.1×.

**The validation idea works.** I simulated the nudge experiment for 8 genres. The model's predicted diversion ratios match the experiment at 0.77 correlation (Netflix: 0.86), and the noise-free truth at 0.99. About a quarter of the viewing a genre lost went to "watch nothing". The rest moved to other titles.

## What didn't

**1. The leftover 10% hides in the taste model.** I gave the same method the *true* tastes, and it landed close to the true bonus (1.02 vs 1.10). The bias comes from rebuilding taste from viewing history. My first, simpler history encoder was off by 24%. A better one, by 11%. So the assumption can hold perfectly in the world and you still get bias, if your model can't see everything the recommender saw.

**2. The error bars lied.** Netflix reports no uncertainty. So I added the standard thing: refit on 10 bootstrap samples, report a 90% interval. All three intervals missed the truth. The model got the ranking right: random is worst, matrix factorization best. But it flipped the sign on popularity (+0.8% predicted, −0.4% true). Bootstrap measures sampling noise. It doesn't measure model bias. A tight interval around a biased estimate looks rigorous and still misleads.

**3. Break the assumption and it's back to the dashboard.** I added one hidden signal: the recommender knows something about you that the model doesn't (think: a search you just did). The model's answer went from 3.4× to **7.0×**. That's almost the naive number, now with a fancy model behind it.

## What I'd take away

1. **Never report "recommended vs not" as impact.** It's selection plus effect.
2. **The method is only as good as what your model sees.** List every signal the recommender uses. If your model doesn't have one, your estimate is biased up.
3. **Validate with a nudge experiment.** It's the only check that touches reality.
4. **A bootstrap interval is not a confidence interval on the truth.** Say so when you show one.

---

*Notebook and code: [github.com/ZFayFay/netflix-rec-impact-replication](https://github.com/ZFayFay/netflix-rec-impact-replication). The method belongs to the Netflix authors (Zielnicki, Aridor, Bibaut, Tran, Chou, Kallus). This is an independent educational rebuild.*
