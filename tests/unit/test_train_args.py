from warcraftsim.puffer.tasks import get_task
from warcraftsim.puffer.train import make_parser, parse


def test_task_train_defaults_fill_what_is_not_given():
    ap = make_parser()
    task = get_task("mirror_mix_abil_hp400")
    args = parse(ap, ["--task", task.name], task.train_defaults)
    assert (args.horizon, args.lr, args.minibatch, args.replay_ratio, args.step_seconds) == (16, 0.003, 192, 4.0, 0.5)
    assert "--train.gae_lambda=0.8" in args.extra and "--train.clip_coef=0.3" in args.extra
    # the command line (and sweep options) win
    args = parse(ap, ["--task", task.name, "--horizon=32", "--lr", "0.001", "--train.gae_lambda=0.9"],
                 task.train_defaults)
    assert (args.horizon, args.lr, args.minibatch) == (32, 0.001, 192)
    assert args.extra.count("--train.gae_lambda=0.9") == 1 and "--train.gae_lambda=0.8" not in args.extra
    # tasks without tuned settings keep the parser's defaults
    plain = parse(ap, ["--task", "micro"], get_task("micro").train_defaults)
    assert plain.horizon == ap.get_default("horizon") and plain.extra == []
