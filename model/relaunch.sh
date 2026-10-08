#!/bin/bash
cd /app/conversations/6a1549534ed513da2a9fbd1b/cifar
if flock -n /tmp/cifar.lock true; then
  flock -n /tmp/cifar.lock ./supervisor2.sh >> supervisor2.log 2>&1 &
fi
