#!/bin/bash

# Usage: run.sh --env=ENV --user=USR EXPERIMENT
#
# Process FILE and optionally apply correction to eitheleft-hand side or
# right-hand side.
#
# Arguments:
#   EXPERIMENT     experiment to run
#
# Options:
#   -h --help
#   --env=E     Environment
#   --user=U    User configuration
#

call_experiment() {
    $1
}

# load the environment configuration
environment() {
    source scripts/envs/$1/$2.sh
}

# experiments
source scripts/experiments/lunar_lander.sh
source scripts/experiments/pusher_3dof.sh

# parsing named arguments
source scripts/docopts.sh --auto "$@"

environment ${ARGS['--env']} ${ARGS['--user']}
call_experiment ${ARGS['EXPERIMENT']}