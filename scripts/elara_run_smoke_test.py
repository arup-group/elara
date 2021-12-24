import argparse
import glob
import subprocess
import sys
import tempfile

from datetime import datetime
from pprint import pprint

from colorama import Fore, Style


def parse_args(cmd_args):
    arg_parser = argparse.ArgumentParser(description='Smoke test a set of elara configs')
    arg_parser.add_argument('-d',
                            '--config_directory',
                            help='the path to the directory containing the configs to test',
                            required=True)
    return vars(arg_parser.parse_args(cmd_args))


def print_banner():
    banner = '''
            _____ _                 
            | ____| | __ _ _ __ __ _ 
            |  _| | |/ _` | '__/ _` |
            | |___| | (_| | | | (_| |
            |_____|_|\__,_|_|  \__,_|
 a,  8a
 `8, `8)                            ,adPPRg,
  8)  ]8                        ,ad888888888b
 ,8' ,8'                    ,gPPR888888888888
,8' ,8'                 ,ad8""   `Y888888888P
8)  8)              ,ad8""        (8888888""
8,  8,          ,ad8""            d888""
`8, `8,     ,ad8""            ,ad8""
 `8, `" ,ad8""            ,ad8""
    ,gPPR8b           ,ad8""
   dP:::::Yb      ,ad8""
   8):::::(8  ,ad8""
   Yb:;;;:d888""
    "8ggg8P"
  ____                  _          _____         _   
 / ___| _ __ ___   ___ | | _____  |_   _|__  ___| |_ 
 \___ \| '_ ` _ \ / _ \| |/ / _ \   | |/ _ \/ __| __|
  ___) | | | | | | (_) |   <  __/   | |  __/\__ \ |_ 
 |____/|_| |_| |_|\___/|_|\_\___|   |_|\___||___/\__|
                                 
    '''
    print("{}{}{}".format(Fore.CYAN, banner, Style.RESET_ALL))


def find_configs(directory):
    paths = glob.glob("{}/*.toml".format(directory))
    print("Found {}{}{} toml configs in {}{}{}".format(Fore.YELLOW,
                                                         len(paths),
                                                         Style.RESET_ALL,
                                                         Fore.YELLOW,
                                                         directory,
                                                         Style.RESET_ALL))
    if not paths:
        print("No notebooks to test - our work here is done. Double check the {}{}{} directory if this seems wrong."
              .format(Fore.YELLOW, directory, Style.RESET_ALL))
        sys.exit(0)
    paths.sort()
    return paths


def run_config(config_path, output_directory):
    print("Executing config '{}{}{}'...".format(Fore.YELLOW, config_path, Style.RESET_ALL))
    execute_notebook_cmd = [
        'elara', 'run',
        '"{}"'.format(config_path),
        '--output_directory_override', '"{}"'.format(output_directory)
        ]
    return run_shell_command(execute_notebook_cmd)


def run_shell_command(shell_cmd):
    print(Fore.BLUE + ' '.join(shell_cmd))
    start_time = datetime.now()
    rc = subprocess.call(' '.join(shell_cmd), shell=True)
    running_time = datetime.now() - start_time
    print("{}Shell process return value was {}{}{}".format(Style.RESET_ALL, Fore.YELLOW, rc, Style.RESET_ALL))
    return rc, ' '.join(shell_cmd), running_time


def print_summary(results_dict):
    print("\n-------------------------------------------------------------")
    print("                        Summary")
    print("-------------------------------------------------------------")
    for config, result in results_dict.items():
        short_name = config.split('/')[-1]
        exit_code, duration = result
        colour = Fore.GREEN if exit_code == 0 else Fore.RED
        outcome = "PASSED" if exit_code == 0 else "FAILED"
        print("{}: {}{} in {}{}".format(
            short_name,
            colour, outcome,
            trim_time_delta(duration), Style.RESET_ALL)
        )
    passes = [ret_code for ret_code, time in results_dict.values() if ret_code == 0]
    failures = [ret_code for ret_code, time in results_dict.values() if ret_code != 0]
    print("\n{} failed, {} passed in {}{}{}".format(
        len(failures),
        len(passes),
        Fore.YELLOW,
        trim_time_delta(datetime.now() - start),
        Style.RESET_ALL)
        )
    print("-------------------------------------------------------------\n")


def trim_time_delta(time_delta):
    return str(time_delta).split('.')[0]


if __name__ == '__main__':
    start = datetime.now()
    command_args = parse_args(sys.argv[1:])
    print_banner()
    print("Smoke testing elara configs in {}'{}'{} directory".format(
        Fore.YELLOW,
        command_args['config_directory'],
        Style.RESET_ALL)
    )

    with tempfile.TemporaryDirectory() as output_directory:

        configs = find_configs(command_args['config_directory'])
        pprint(configs, width=120)
        print("")

        run_results = {}
        for config in configs:
            print('------------------------------------------------------')
            return_code, cmd, run_time = run_config(config, output_directory)
            run_results[config] = (return_code, run_time)

    print_summary(run_results)
    sys.exit(sum(ret_code for ret_code, time in run_results.values()))