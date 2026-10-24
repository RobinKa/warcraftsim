# GPU unlock and power limit

Since 2026-10-24 the training machine's RTX 3090 runs with its BF16/FP16 tensor-core rate unlocked ([gpu-unlock](https://github.com/Zagitta/gpu-unlock)) and its power limit at 400 W. Both are setup of this machine, outside the repository. This page has what they do, what they gave, and how to set them up, check them and undo them.

## What it is

GeForce cards of the RTX 30 series issue BF16 and FP16 tensor-core instructions that accumulate into FP32 at half the rate of the professional cards built from the same chips. gpu-unlock is a Linux kernel module for GA102, GA103, GA104 and GA106 chips (the RTX 3090 is a GA102). At every boot, before the NVIDIA driver takes the card, it writes the card's speed selectors to the professional rate with a firmware payload derived from NVIDIA's CMP cards. The NVIDIA driver itself is not patched. Plain FP32 math is not affected.

The learner's updates and the actors' inference run their matrix multiplies in BF16 (`torch.autocast`), so both can use the faster rate.

## What it gave

Matrix multiplies, `scripts/bench_gemm.py` (TFLOPS; the files are in `runs/bench/`). The README's size is 8192³; the others are the shapes of our model's layers over a minibatch's entity tokens.

| | before, 350 W | before, 400 W | unlocked, 350 W | unlocked, 400 W |
|---|---|---|---|---|
| BF16 8192³ | 65.6 | 73.3 | 115.5 | 121.9 |
| FP16 8192³ | 65.4 | 72.2 | 111.8 | 118.8 |
| TF32 8192³ | 35.4 | 38.4 | 62.5 | — |
| FP32 8192³ | 22.9 | 25.6 | 24.6 | — |
| BF16 65536×256×1024 | 63.5 | 65.8 | 87.9 | 93.5 |
| BF16 65536×1024×256 | 63.9 | 67.8 | 97.6 | 103.3 |
| BF16 16384×256×256 | 44.9 | 46.9 | 62.0 | 63.8 |

The unlock makes BF16 1.76 times faster at the README's size and 1.4 to 1.5 times at ours. The higher power limit adds 3 to 7% on top.

Self-play's speed is set mostly by the games, which run on the CPU ([optimizations](../optimizations.md)): before the unlock the learner spent ~2.7 s an update computing and ~2.9 s waiting for the games' steps. The training run's steps per second before and after are under [the effect on training](#the-effect-on-training).

## The effect on training

To be measured on `fgself-139` (resumed at 11:00 on 2026-10-24 with both), against its own speed before the reboot: ~1,700-1,800 agent steps per second.

## Setup on this machine

Ubuntu 26.04, kernel 7.0.0-38, NVIDIA driver 595.99, the card at PCI address `0000:07:00.0`, Secure Boot off.

1. Install the module and its firmware (two Debian packages, `gpu-unlock-dkms` and `gpu-unlock-firmware`; DKMS builds the module for each kernel with headers):

   ```
   git clone https://github.com/Zagitta/gpu-unlock ~/gpu-unlock
   cd ~/gpu-unlock && make install
   ```

2. This Ubuntu builds its initramfs with dracut (`update-initramfs` is a dracut wrapper), but `make install` saw Ubuntu and put in only the initramfs-tools config, which dracut ignores: after the first reboot the module had not loaded. Add the repository's dracut config and rebuild the initramfs:

   ```
   sudo make -C ~/gpu-unlock install-dracut     # /usr/lib/dracut/dracut.conf.d/gpu-unlock.conf: force_drivers+=" gpu_unlock "
   sudo update-initramfs -u -k $(uname -r)
   sudo lsinitramfs /boot/initrd.img-$(uname -r) | grep gpu.unlock   # the module and its five firmware files
   ```

3. Name the card on the kernel command line, in `/etc/default/grub`:

   ```
   GRUB_CMDLINE_LINUX_DEFAULT="gpu_unlock.targets=0000:07:00.0"
   ```

   then `sudo update-grub` and reboot. The screen stays blank for a moment early in the boot while the module resets the card.

4. The power limit: the card's default is 350 W and its firmware allows up to 400 W (`nvidia-smi -q -d POWER`). `nvidia-smi -pl` lasts until a reboot or a suspend, so a systemd unit sets it at boot and after every resume, `/etc/systemd/system/nvidia-power-limit.service`:

   ```
   [Unit]
   Description=RTX 3090 power limit 400 W (its firmware's maximum; the default is 350 W)
   After=nvidia-persistenced.service suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target
   Wants=nvidia-persistenced.service

   [Service]
   Type=oneshot
   ExecStart=/usr/bin/nvidia-smi -i 0000:07:00.0 -pl 400
   RemainAfterExit=yes

   [Install]
   WantedBy=multi-user.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target
   ```

   ```
   sudo systemctl daemon-reload && sudo systemctl enable --now nvidia-power-limit.service
   ```

   Under training the card drew 260-340 W at ~60 °C before the unlock. With 400 W on the card, the power supply should be a good 850 W or more (an RTX 3090 draws short spikes well above its limit).

## Checking it

```
sudo journalctl -k -b | grep gpu_unlock      # ends with "pci 0000:07:00.0: gpu_unlock: unlocked"
lsmod | grep gpu_unlock
nvidia-smi --query-gpu=power.limit --format=csv     # 400.00 W
python scripts/bench_gemm.py --out runs/bench/check.json --compare runs/bench/gpu-after-unlock-400w.json
```

The kernel notes the module as out of tree and unsigned ("tainting kernel"); that is expected with Secure Boot off.

## Caveats

* **A kernel update.** DKMS builds the module for the new kernel; check that the new initramfs has it (`sudo lsinitramfs /boot/initrd.img-<version> | grep gpu.unlock`), and if not, `sudo update-initramfs -u -k <version>`.
* **Suspend** powers the card off and resets its speed selectors: the module unlocks it again on resume (about a second; `gpu_unlock.resume=0` turns that off), and the power limit unit runs again after the resume. Hibernation is not covered by the module.
* **A driver update** needs nothing: the module runs before the unmodified NVIDIA driver.
* The project is young and has one author; its README names no tested driver versions and makes no stability claims. If the card's firmware ever fails to stop, the README says to power off before trying again.

## Undoing it

1. Remove `gpu_unlock.targets=...` from `/etc/default/grub` and run `sudo update-grub`.
2. `sudo rm /usr/lib/dracut/dracut.conf.d/gpu-unlock.conf` (not owned by the packages), `sudo apt remove gpu-unlock-dkms gpu-unlock-firmware`, `sudo update-initramfs -u`.
3. `sudo systemctl disable --now nvidia-power-limit.service`, `sudo rm /etc/systemd/system/nvidia-power-limit.service`, `sudo nvidia-smi -pl 350`.
4. Reboot.
