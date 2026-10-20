# Ideas

Where the project's ideas came from and what became of them. Many are the user's; others come from AlphaStar and OpenAI Five ([what those papers found](ideas-from-alphastar-and-openai-five.md)), from LLM reinforcement learning, from image generation, or from our own runs. Each experiment's numbers are in the [experiment log](experiment-log.md).

## The queue (2026-10-20)

In the order they run. The code for each is in place unless it says otherwise.

| idea | from | what it is | status |
|---|---|---|---|
| What the agent can't see or do | the user's question | each hero's abilities, cooldowns and items, what buildings make, the clock, the enemies it saw before (`--inputs 3`); masks for orders the game refuses (`--tech-mask`, `--valid-mask`) | running: `fgself-107` |
| Narrow target strategies | the user ("they all seem big and generic") | a game's heroes, its main army and what they need; the rest free; drawn evenly per signature (`strategy.narrow`, `--narrow` pools) | next |
| More kinds of strategy conditioning | the user ("no rush" and many more); AlphaStar's z had statistics beside the build order | no rush, rush, expand, no expansion, workers, creeping, harass, scouting, army size, low upkeep: a switch and a parameter each (`fullgame/modes.py`, `--z-modes`) | next |
| Rewards by part | the user (rewards combined in LLM RL): GDPO, SAW | an advantage per reward (outcome, shaping, strategy), each normalized, a saturated one held small (`--reward-parts`) | built |
| Neutral shops, the tavern, mercenaries | the user's question | a `Buy` order (checked in the real game: a worker hires a tavern hero, a hero buys an item, a unit hires a mercenary); new order classes, the vocabulary and the network grown | the harness done; the rest to build |
| A heuristic teacher | the user ("buying a first hero with a worker": what it rarely stumbles on) | our own advisor whose orders label the policy's steps, as the target builder does (`--z-build 2`): tavern heroes, a town portal, potions, a portal home, items on the ground | to build |
| Self-distillation with privileged information (OPSD) | the user, from LLM training | the same network given what the player can't see is the teacher; a KL pulls the policy without it toward it; never in evaluation | to build |
| Scenarios and weight averaging | the user (the micro curriculum, kept on); model soups, task arithmetic | mid-game situations the harness sets up; specialists forked from one checkpoint, merged by averaging their weights | to build |
| Human data | AlphaStar (Fig. 3E, its largest gain) | a cloning loss on people's 95 Echo Isles games, played back in our client | later |
| Fewer decisions, spatial point heads | AlphaStar (action rate), SC2LE | a "keep this order k steps" head; points on a map, not x then y | later |

## Tried

| idea | from | result |
|---|---|---|
| Strategic variety first, a lower win rate accepted for a while | the user (2026-10-18) | the direction since; targets the policy follows on its own came with the tech-tree builder (`fgself-95`) |
| Faster games for variety: the duel map, faster rules | the user | `echofast` training judged on the real game: the best line (`fgself-100`, 95%) |
| GRPO from mid-game branches | the user (from LLM RL: DeepSeekMath's group-relative advantages) | level (`fgself-96`, `-97`): only ~55% of branches replay exactly (creeps and the AI are not deterministic) |
| Classifier-free guidance toward the target | the user (from image generation) | no use (`fgself-106`): it followed no better and won 50% instead of 91% |
| AlphaStar's league (self-play, past snapshots, a main exploiter) | AlphaStar | level (`fgself-103`) |
| Reward decay 0.6^(T / 10 min) | OpenAI Five (App. G, eq. 11) | level (`fgself-104`) |
| Each sample used once | OpenAI Five (§4.4) | level (`fgself-105`) |
| More games as the weaker races | ours | undead 79% → 89% (`fgself-102`) |
| The anchor at the best checkpoint | ours | no slide after the peak (`fgself-101`) |
| Forward KL on every head, UPGO, a moving anchor, a value that sees the opponent | AlphaStar | forward KL and UPGO: large gains; the opponent's information for the value: kept, level |
| A larger network | ours | 5 layers helped on the duel maps; 8 level, and the user's direction: no more layers without a clear gain |

## Sources

* AlphaStar, OpenAI Five, SC2LE: see [what the papers found](ideas-from-alphastar-and-openai-five.md).
* GDPO (NVIDIA): rewards normalized one by one before they are summed ([explained](https://aipapersacademy.com/gdpo/)). SAW: stage-aware dynamic weighting, [arXiv 2606.07705](https://arxiv.org/abs/2606.07705).
* GRPO: Z. Shao et al., DeepSeekMath, [arXiv 2402.03300](https://arxiv.org/abs/2402.03300).
* Classifier-free guidance: J. Ho and T. Salimans, [arXiv 2207.12598](https://arxiv.org/abs/2207.12598).
* Model soups: M. Wortsman et al., [arXiv 2203.05482](https://arxiv.org/abs/2203.05482). Task arithmetic: G. Ilharco et al., [arXiv 2212.04089](https://arxiv.org/abs/2212.04089).
