import argparse
import os
from datetime import timedelta
from subprocess import Popen, PIPE
import time

EXPERIMENT_TEMPLATE = 'nohup python -m experiments.domainrand.{main} {setting} --experiment-prefix={prefix} ' \
                      '--seed={seed} {extra}'
SLEEP_TIME = 30  # in seconds

LOG_DIRECTORY = 'tails'
LOG_FILENAME_TEMPLATE = '{dir}/{main}-{setting}-{prefix}-{seed}.log'


def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--main')
    parser.add_argument('--setting', choices=['lunar', 'pusher', 'ergo'])
    parser.add_argument('--prefix')
    parser.add_argument('--first-seed', default=0, type=int)
    parser.add_argument('--seeds', type=int)

    parser.add_argument('with', choices=['with'])
    parser.add_argument('extra', nargs=argparse.REMAINDER)

    return parser.parse_args()


def call_module(main, setting, prefix, seed, arguments):
    cmd = EXPERIMENT_TEMPLATE.format(
        main=main,
        setting=setting,
        prefix=prefix,
        seed=seed,
        extra=" ".join(arguments)
    )
    args = cmd.split(' ')

    log_filename = LOG_FILENAME_TEMPLATE.format(
        dir=LOG_DIRECTORY,
        main=main,
        setting=setting,
        prefix=prefix,
        seed=seed
    )
    log_file = open(log_filename, '+w')

    handler = Popen(args=args, stdin=log_file, stdout=log_file, stderr=log_file)

    return handler


def call_multi_seed(main, setting, prefix, initial_seed, seeds, extra):
    process_handlers = []

    if not os.path.exists(LOG_DIRECTORY):
        os.mkdir(LOG_DIRECTORY)

    for index in range(seeds):
        handler = call_module(
            main=main,
            setting=setting,
            prefix=prefix,
            seed=index + initial_seed,
            arguments=extra
        )
        process_handlers.append(handler)
    print("{} Seeds with PID = [{}]".format(seeds, ", ".join(list(map(lambda p: str(p.pid), process_handlers)))))
    return process_handlers


def is_process_running(p):
    return p.poll() is None


def wait_all(process_handlers):
    _time = time.time()

    while any(map(is_process_running, process_handlers)):
        print('\rWaiting for all seeds to finish...', end='')
        time.sleep(SLEEP_TIME)
    _time = time.time() - _time - SLEEP_TIME

    return _time


def exit_status(process_handlers):
    return list(map(lambda p: str(p.poll()), process_handlers))


def run_experiment(args):
    print('Experiments')
    print('===================================')
    print("Launching experiment <{experiment}> with <{setting}>.".format(experiment=args.main,
                                                                         setting=args.setting))
    process_handlers = call_multi_seed(
        main=args.main,
        setting=args.setting,
        prefix=args.prefix,
        initial_seed=args.first_seed,
        seeds=args.seeds,
        extra=args.extra
    )
    _time = wait_all(process_handlers)
    print()
    print('<-------- COMPLETED -------------->')
    seeds_status = exit_status(process_handlers)
    print('Seeds Exit Status = [{}]'.format(",".join(seeds_status)))
    print('Elapsed Time = {}'.format(str(timedelta(seconds=_time))))
    print('===================================')

    return all(int(status) == 0 for status in seeds_status)  # if all exit statuses r 0


def collect_data(args):
    print()
    print('Data Collection')
    print('===================================')
    print("Launching data recollection of <{experiment}> with <{setting}>.".format(experiment=args.main,
                                                                                   setting=args.setting))
    process_handlers = call_multi_seed(
        main='batch_reward_analysis',
        setting=args.setting,
        prefix=args.prefix,
        initial_seed=args.first_seed,
        seeds=args.seeds,
        extra=args.extra
    )
    _time = wait_all(process_handlers)
    print()
    seeds_status = exit_status(process_handlers)
    print('<-------- COMPLETED -------------->')
    print('Seeds Exit Status = [{}]'.format(",".join(seeds_status)))
    print('Elapsed Time = {}'.format(str(timedelta(seconds=_time))))
    print('===================================')

    return all(int(status) == 0 for status in seeds_status)  # if all exit statuses r 0


def launch():

    print('<---- RUNNING ---->')

    args = parse()

    steps = [
        run_experiment,
        collect_data
    ]

    done = False
    for step in steps:
        done = step(args)
        if not done:
            break
    print()
    print('<---- DONE: {} --->'.format(done))


if __name__ == '__main__':
    launch()
