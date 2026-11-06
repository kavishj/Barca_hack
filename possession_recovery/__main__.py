import argparse


def main():
    parser=argparse.ArgumentParser(description='Contact-aware short-possession recovery')
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare')
    for name in ('graph','rich-features','out'):prepare.add_argument('--'+name,required=True)
    compare=sub.add_parser('compare')
    for name in ('features','audio','graph','source-study','train-labels','validation-labels','out'):
        compare.add_argument('--'+name,required=True)
    compare.add_argument('--epochs',type=int,default=60)
    compare.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    compare.add_argument('--device',default='cuda',choices=('cpu','cuda'))
    compare.add_argument('--bootstrap',type=int,default=1000)
    predict=sub.add_parser('predict')
    for name in ('features','audio','graph','model','out'):predict.add_argument('--'+name,required=True)
    predict.add_argument('--device',default='cuda',choices=('cpu','cuda'))
    args=parser.parse_args()
    if args.command=='prepare':
        from .features import prepare as run
    elif args.command=='compare':
        from .experiment import compare as run
    else:
        from .experiment import predict as run
    run(args)


if __name__=='__main__':main()
