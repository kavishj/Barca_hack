import argparse
from .experiment import compare,predict
parser=argparse.ArgumentParser()
commands=parser.add_subparsers(dest='command',required=True)
for command in ('compare','predict'):
    p=commands.add_parser(command)
    for name in ('features','audio','graph','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu')
    if command=='compare':
        for name in ('source-study','boosted-study','train-labels','validation-labels','pretrained'):p.add_argument('--'+name,required=True)
        p.add_argument('--epochs',type=int,default=60);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
        p.add_argument('--bootstrap',type=int,default=1000)
    else:p.add_argument('--model',required=True)
args=parser.parse_args()
compare(args) if args.command=='compare' else predict(args)
