#!/usr/bin/env bash

echo "configuring -> Bluewire Environment"

source `which virtualenvwrapper.sh`
workon diffsim
export PYTHONPATH="${PYTHONPATH}:`pwd`"
Xvfb :1 -screen 0 84x84x24 -ac +extension GLX +render -noreset &> xvfb.log &
export DISPLAY=:1
