import pandas as pd
import argparse
import os
from pathlib import Path
import time
import datetime
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save


NUM_EXPERIMENTS = 5
DATASETS = [
    "ERing",
    "Libras",
    "AtrialFibrillation",
    "BasicMotions",
    "RacketSports",
    "Handwriting",
    "Epilepsy",
    "JapaneseVowels",
    "UWaveGestureLibrary",
    "PenDigits",
    "StandWalkJump",
    "NATOPS",
    "ArticularyWordRecognition",
    "FingerMovements",
    "LSST",
    "HandMovementDirection",
    "Cricket",
    "CharacterTrajectories",
    "EthanolConcentration",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "Heartbeat",
    "PhonemeSpectra",
    "SpokenArabicDigits",
    "EigenWorms",
    "DuckDuckGeese",
    "PEMS-SF",
    "FaceDetection",
    "MotorImagery",
    "InsectWingbeat",
]


def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--depth', type=int, default=10, help='Number of dilated convolution on encoder')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    args = parser.parse_args()

    exp_dir = Path('training/' + name_with_datetime(args.run_name+"_UEA"))


    results_data_dir = {
        'classifier': [],
        'run_name': [],
        'dataset': [],
        'exp': [],
        'acc': [],
        'f1': [],
        'recall': [],
        'precision': [],
        'time': [],
        'auprc': []
    }

    
    for dataset in DATASETS:
        print("Dataset:", dataset)
        print("Arguments:", str(args))
        
        device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
        
        print('Loading data... ', end='')
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(dataset)
        print('done')
        
        config = dict(
            batch_size=args.batch_size,
            lr=args.lr,
            output_dims=args.repr_dims,
            max_train_length=args.max_train_length,
            depth=args.depth
        )

        for i in range(NUM_EXPERIMENTS):

            run_dir = exp_dir / f"{dataset}_run{i}"
            os.makedirs(run_dir, exist_ok=True)
            
            t = time.time()
            
            model = TS2Vec(
                input_dims=train_data.shape[-1],
                device=device,
                **config
            )
            loss_log = model.fit(
                train_data,
                n_epochs=args.epochs,
                n_iters=args.iters,
                verbose=True
            )
            model.save(f'{run_dir}/model.pt')

            t = time.time() - t
            print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

            if args.eval:
                out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
                pkl_save(f'{run_dir}/out.pkl', out)
                pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
                print('Evaluation result:', eval_res)


            results_data_dir['classifier'].append('svm')
            results_data_dir['run_name'].append(args.run_name)
            results_data_dir['dataset'].append(dataset)
            results_data_dir['exp'].append(i)
            results_data_dir['acc'].append(eval_res['acc'])
            results_data_dir['f1'].append(eval_res['f1'])
            results_data_dir['recall'].append(eval_res['recall'])
            results_data_dir['precision'].append(eval_res['precision'])
            results_data_dir['time'].append(t)
            results_data_dir['auprc'].append(eval_res['auprc'])

            results_df = pd.DataFrame(results_data_dir)
            results_df.to_csv(exp_dir / 'results.csv', index=False)

    print("Finished.")
