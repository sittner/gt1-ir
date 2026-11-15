#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-only
# install gt1conv from this directory and restart the service
set -e
cd "$(dirname "$0")"
make -s gt1conv
sudo install -m 755 gt1conv /usr/local/bin/
sudo install -m 644 gt1conv.service /etc/systemd/system/
sudo install -m 644 gt1conv.default /etc/default/gt1conv
[ -f /home/pi/gt1conv.conf ] || cp gt1conv.conf /home/pi/gt1conv.conf
sudo systemctl daemon-reload
sudo systemctl enable gt1conv >/dev/null 2>&1
sudo systemctl restart gt1conv
sleep 8
journalctl -u gt1conv -o cat --no-pager -n 12
