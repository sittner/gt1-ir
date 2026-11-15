# gt1conv – low-latency IR convolver for the BOSS GT-1

`gt1conv` turns a Raspberry Pi into a cabinet-IR convolver for the BOSS GT-1, using the GT-1's USB
audio interface as an effects loop:

```
guitar -> GT-1 -> USB capture -> gt1conv (IR convolution) -> USB playback -> GT-1 output
```

The GT-1 is also used as MIDI controller: the IR follows the selected patch (mapping by patch number,
or by a token in the patch name that can be edited directly on the device).

## Features

- **~3.6 ms round trip** (analog in -> analog out, measured) at 44.1 kHz, period 8 / fill 24 frames
- **zero algorithmic latency**: first 256 IR taps as direct FIR in the audio thread, the rest as
  non-uniformly partitioned FFT convolution (block 128 / 1024 / 8192) in worker threads
- single clock domain (GT-1 USB), no resampling or drift compensation needed
- **IR per patch**: `U01..U99`, `P01..P99`, name tokens `IR0..IR9`, `IRa..IRz`, `IRA..IRZ`, default, bypass
- glitch-free IR switching: the new engine is primed with the input history, warmed up and crossfaded
- IR loading: WAV 8/16/24/32-bit int, 32/64-bit float, mono/stereo, any sample rate (resampled to 44.1 kHz)
- level normalization (loudest 1/3-octave band in the guitar range), soft clipper
- config file reloaded automatically, systemd service, recovers from GT-1 power cycles
- built-in selftest against brute-force convolution

## Hardware / GT-1 setup

- Raspberry Pi 3B+ (or better), Raspberry Pi OS / Debian trixie, GT-1 connected via USB
- GT-1: **USB > Direct Mon = OFF**. The GT-1 switches Direct Mon on at every power-up and the setting
  is not accessible via MIDI, so this is a manual step after powering on the GT-1.
- Use the GT-1 patches without its own speaker simulation (otherwise two cabinets are in series).
- Note: the USB send of the GT-1 is the end of its effects chain, so the IR comes after delay / reverb.

## Build and install (on the Pi)

```sh
sudo apt install build-essential libasound2-dev libfftw3-dev
make
./install.sh        # installs binary + systemd unit, copies the example config if none exists
journalctl -u gt1conv -f
```

`install.sh` installs `/usr/local/bin/gt1conv`, `/etc/systemd/system/gt1conv.service`,
`/etc/default/gt1conv` (audio options) and, if missing, `/home/pi/gt1conv.conf`.
The first start plans the FFTs (can take ~2 minutes on a Pi 3); the result is cached in
`/var/tmp/gt1conv.wisdom`.

## Configuration

`/home/pi/gt1conv.conf` (see [gt1conv.conf](gt1conv.conf) for all options). Changes are applied within
about a second; a config with errors is rejected and the previous one stays active.

```ini
irdir   = /home/pi/ir
norm    = peak              # loudest 1/3-octave band 80 Hz..6 kHz set to 'ref' dB
ref     = 1.5               # matches the level of the GT-1 cab sim
default = "Magma Vintage 1x12 Medium Mix.wav"
U05     = bypass
P12     = V7X_dc.wav gain -2
IR0     = bypass
IR1     = "EV MIX B.wav"
```

Priority: **name token** (e.g. patch name `CLEAN TWIN IR1`; token at the start or after a space,
followed by a space or the end) **> patch number > default**.

Audio options in `/etc/default/gt1conv`: `-p` period (default 8), `-f` playback fill (24 recommended
on a Pi 3), `-m` mono (convolve L+R once, half CPU), `-s` statistics interval.

Manual run with a single IR (no MIDI): `sudo ./gt1conv -i ir.wav -g 0`

Selftest: `./gt1conv -S -i ir.wav`

## How the GT-1 is controlled

Via USB MIDI (Roland SysEx, model id `00 00 00 30`):

- editor mode on/off: DT1 `7F 00 00 01` = `01` / `00` (as the BOSS editor does); in editor mode the
  GT-1 reports patch changes as DT1 `00 01 00 00 [hi lo]`
- current (temporary) patch name: RQ1 `60 00 00 00`, stored names: `10 nn 00 00` (user),
  `20 nn 00 00` (preset); used to identify the current patch at startup and to read name tokens

gt1conv only reads from the GT-1 apart from switching editor mode. Address map reference:
[GT-1 FxFloorBoard](https://sourceforge.net/projects/fxfloorboard/files/GT-1FxFloorBoard/) `midi.xml`.

## Latency measurements (Pi 3B+, GT-1 out -> GT-1 in loopback)

| path | period / fill | round trip |
|---|---|---|
| GT-1 USB loop | 8 / 16 | 2.4 – 3.4 ms (startup dependent) |
| GT-1 USB loop | 8 / 24 | ~3.6 ms |
| GT-1 USB loop | 16 / 32 | 5.4 ms |
| GT-1 in -> HiFiBerry DAC+ out | 32 / 96 | 3.6 ms (two clocks, 29 ppm drift) |

`latency.c` is the loopback tester used for these numbers (needs a cable GT-1 output -> GT-1 input,
Direct Mon off): `sudo PB=hw:GT1 ./latency 8 24 10`

## Files

| file | |
|---|---|
| `gt1conv.c` | convolver |
| `gt1conv.conf` | example configuration |
| `gt1conv.service`, `gt1conv.default` | systemd unit, audio options |
| `install.sh` | build + install + restart on the Pi |
| `latency.c` | loopback latency tester |

## License

GNU General Public License v2.0, see [LICENSE](LICENSE).
