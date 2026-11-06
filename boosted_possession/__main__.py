import argparse
from .experiment import compare,regenerate

parser=argparse.ArgumentParser(description=__doc__)
commands=parser.add_subparsers(dest='command',required=True)
for command in ('compare','predict'):
    p=commands.add_parser(command)
    for name in ('features','rich-features','audio','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--threads',type=int,default=4)
    if command=='compare':
        for name in ('source-study','train-labels','validation-labels'):p.add_argument('--'+name,required=True)
        p.add_argument('--bootstrap',type=int,default=1000)
    else:p.add_argument('--model',required=True)
args=parser.parse_args()
compare(args) if args.command=='compare' else regenerate(args)
