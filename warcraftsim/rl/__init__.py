"""Reinforcement learning with our own PyTorch trainer (entity networks, pointer heads, league self-play).

The games run in the same bridge workers as for PufferLib (warcraftsim.puffer.bridge); the trainer
talks to them with the bridge's socket protocol (client.py). train.py --trainer torch launches it;
it runs with a Python that has torch (WC3_TORCH_PYTHON), like the behavior cloning fit, and needs
only numpy and torch: the task is described to it by a JSON spec (spaces, heads, masks).
"""
