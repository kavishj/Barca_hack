import argparse


def main():
    parser=argparse.ArgumentParser(description='Relational possession prepare|compare|predict')
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare')
    for name in ('data-dir','out'):
        prepare.add_argument('--'+name,required=True)
    compare=sub.add_parser('compare')
    for name in ('features','audio','graph','train-labels','validation-labels','audit','out'):
        compare.add_argument('--'+name,required=True)
    compare.add_argument('--epochs',type=int,default=60)
    compare.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    compare.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    compare.add_argument('--bootstrap',type=int,default=1000)
    compare.add_argument('--compare',nargs='*',default=[])
    compare.add_argument('--cross-validate',action='store_true')
    compare.add_argument('--calibrate',action='store_true')
    predict=sub.add_parser('predict')
    for name in ('features','audio','graph','model','out'):
        predict.add_argument('--'+name,required=True)
    predict.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    predict.add_argument('--calibrated',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        from .data import prepare as run
    elif args.command=='compare':
        from .experiment import compare as run
    else:
        from .experiment import predict as run
    run(args)


if __name__=='__main__':
    main()
