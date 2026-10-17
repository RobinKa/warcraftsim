# Reward shaping (Ng, Harada and Russell, 1999)

**The paper:** A. Y. Ng, D. Harada and S. Russell, "Policy invariance under reward transformations: Theory and application to reward shaping", *ICML* 1999. [PDF](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf) ([other copy](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)).

Shaping adds rewards that guide an agent before it reaches the real goal. The paper shows which extra rewards are safe: those that are a difference of a "potential". Every other kind can teach the agent to do the wrong thing. Our material reward in the whole game is this kind of shaping.

## The problem: extra rewards can change the goal

The paper starts with two bugs from other projects (§1):

* **A bicycle** (Randløv and Alstrøm, 1998): it got a reward for each step towards the goal and no penalty for a step away. The agent learned to ride in tiny circles near the start. Each circle had a part that moved towards the goal, and that part paid.
* **A soccer robot:** it got a reward for touching the ball. It learned to stay next to the ball and "vibrate", to touch it as often as possible.

In both cases a loop of states paid more than the task itself.

## The fix: rewards from a potential

Choose any function Φ of the state: the **potential**. Give the shaping reward

> F(s → s′) = γ · Φ(s′) − Φ(s)

for every step from state s to state s′ (γ is the discount).

**Theorem 1** (§3): shaping of this form never changes which policies are optimal. And without more knowledge of the task, it is the only form with that guarantee: for any other form there are tasks where it makes the agent learn a wrong policy.

**Why loops cannot pay:** with γ = 1, the rewards of a loop s₁ → s₂ → … → s₁ add up to Φ(s₁) − Φ(s₁) = 0. The bicycle's circles would earn nothing.

## An example: a corridor

Five cells in a row. The agent starts at the left and gets +1 when it reaches G. No discount (γ = 1). The potential is minus 0.1 per step still to go, and 0 at the end:

| cell | S0 | S1 | S2 | S3 | G |
|---|---|---|---|---|---|
| Φ | −0.4 | −0.3 | −0.2 | −0.1 | 0 |

* **A step forward** pays Φ(next) − Φ(now) = +0.1. **A step back** pays −0.1.
* **Back and forth** S1 → S2 → S1 pays +0.1 − 0.1 = 0. Walking in circles earns nothing.
* **Every path** from S0 to G collects the same shaping in total: Φ(G) − Φ(S0) = +0.4. So the shaping cannot make one path better than another. It only tells the agent sooner that forward is good.

**The bad version:** pay +0.1 for a step forward and 0 for a step back. Now S1 → S2 → S1 → S2 → … earns +0.1 every two steps, without end. That beats the +1 at the goal: this is the bicycle bug.

## What shaping does to the value

**Corollary 2** (§3): with potential shaping, the values become

> V′(s) = V(s) − Φ(s)  and  Q′(s, a) = Q(s, a) − Φ(s)

where V is the value without shaping (Φ is 0 at the end of an episode). Remark 1 adds that this holds for every policy, not only the optimal one.

In the corridor: from S0 the best path earns 4 × 0.1 + 1 = 1.4 = 1 − (−0.4). From S2 it earns 2 × 0.1 + 1 = 1.2 = 1 − (−0.2).

**Why the minus:** the value counts only the rewards still to come. Shaping pays out the potential on the way, so what is still to come is the old value less the potential. A state with a high potential has already been paid for it. It is not a worse state.

All Q-values of a state move by the same −Φ(s). So the best action does not change, and that is the whole proof of Theorem 1.

## Choosing the potential

* **The best potential is the value itself:** with Φ = V*, every shaped value is 0. The agent then only has to learn which actions are better than others (§3).
* **A rough estimate also helps** (§4). In grid worlds with an 80% chance to move as intended, the authors used Φ = −(Manhattan distance to the goal) / 0.8: about the steps still needed. Learning (Sarsa, 40 runs) was much faster on a 10 × 10 grid. On a 50 × 50 grid, learning without shaping fell far behind. Half that potential, far from the true value, also helped a lot.
* **Subgoals:** in a 5 × 5 grid with flags to collect in order, a potential that rose with each flag collected also sped up learning.

## In warcraftsim

The whole-game reward (`fullgame/selfplay.py`) is +1 for a win and −1 for a loss, plus potential shaping:

* **The potential** is the material lead: what each side's living units and buildings cost, less half the gold and lumber it holds (`--float-penalty 0.5`: spending is credited at the order). It is in units of 2,000 (`--shaping-scale`) and weighted by `--shaping` (0.5 now).
* **Each step** pays γ · Φ(next) − Φ(now), with γ = 0.9997.
* **The last step** subtracts the potential, so Φ is 0 at the end, as the theorem needs. Over a game the shaping adds up to about 0. Only the result counts in total.

**The value chart:** by Corollary 2 the learned value is the expected result minus 0.5 × the material lead. A lead makes the value read low, and a big deficit makes it read high. In one lost undead game the value was −0.52 at minute 5, with the learner ahead by 1.94. It was +1.61 at the end, with the learner behind by 5.0. The video's panel therefore shows the **win estimate**: value + 0.5 × lead. In that game it read +0.45 at minute 5 and −0.92 at the end.

## Where theory and practice differ

The theorem is about optimal policies and exact values. Real training has neither:

* **The value lags.** The agent learns with an estimated value and short-horizon credit (GAE). A shaping reward arrives at once. The value term that cancels it arrives only as fast as the value learns. Material that lasts (workers, buildings) can therefore look better than it is. In long games our learner makes ~33 workers against the AI's ~19.
* **The weight matters in practice.** Halving the shaping (1.0 → 0.5) was one of our largest gains: 62% → 72% of games against insane (`fgself-73`). A quarter (0.25) was level.
* **A next test:** count workers in the potential only up to a cap. If the worker pile-up comes from the shaping, the cap removes it.

See also: [OpenAI Five](openai-five.md), which shaped with rewards for kills, gold and experience that shrink over a game, and [AlphaStar](alphastar.md), which used no material shaping.
