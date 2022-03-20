# use the trained model to distillation knowledge from teacher
import argparse

from args import train_argparser, eval_argparser
from config_reader import process_configs
from spert import input_reader
from spert.ster_trainer import STERTrainer


def __train(run_args):
    trainer = STERTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader, input_reader_cls_TeaE=input_reader.JsonInputReaderTeaE, input_reader_cls_TeaR=input_reader.JsonInputReaderTeaR)


def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __eval(run_args):
    trainer = STERTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('--mode', type=str, default="eval", help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()

    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    else:
        raise Exception("Mode not in ['train', 'eval'], e.g. 'python spert.py train ...'")
