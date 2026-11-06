import argparse


def main():
    p=argparse.ArgumentParser(description='Frozen visual possession pilot and experiments')
    sub=p.add_subparsers(dest='command',required=True)
    pilot=sub.add_parser('pilot')
    for name in ('video','model-cache','out'):pilot.add_argument('--'+name,required=True)
    prepare=sub.add_parser('prepare')
    for name in ('data-dir','video','out'):prepare.add_argument('--'+name,required=True)
    prepare.add_argument('--step',type=int,default=50)
    extract=sub.add_parser('extract')
    for name in ('video','geometry','model-cache','out'):extract.add_argument('--'+name,required=True)
    extract.add_argument('--intervals')
    worker=sub.add_parser('worker')
    for name in ('video','geometry','model-cache','out','selection','devices'):worker.add_argument('--'+name,required=True)
    compare=sub.add_parser('compare')
    for name in ('features','audio','graph','vision','train-labels','validation-labels','source-study','out'):
        compare.add_argument('--'+name,required=True)
    compare.add_argument('--epochs',type=int,default=60)
    compare.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    compare.add_argument('--device',default='cuda',choices=('cpu','cuda'))
    compare.add_argument('--bootstrap',type=int,default=1000)
    predict=sub.add_parser('predict')
    for name in ('features','audio','graph','vision','model','out'):predict.add_argument('--'+name,required=True)
    predict.add_argument('--device',default='cuda',choices=('cpu','cuda'))
    predict.add_argument('--calibrated',action='store_true')
    args=p.parse_args()
    if args.command=='pilot':
        from .internvideo import pilot as run
    elif args.command=='prepare':
        from .geometry import prepare as run
    elif args.command=='extract':
        from .extract import extract as run
    elif args.command=='worker':
        from .extract import worker as run
    elif args.command=='compare':
        from .experiment import compare as run
    else:
        from .experiment import predict as run
    run(args)


if __name__=='__main__':main()
