# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MODIFIED by Reka Cserhati on 28/06/2021: argparse + non-default embeddings + save_translated()


import os
import multiprocessing
import numpy as np
from icp import ICPTrainer
import utils
import time
import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("src_lang", help="language ID of the source language")
parser.add_argument("tgt_lang", help="language ID of the target language")
parser.add_argument("--src_emb", help="path to the embedding file in the source language")
parser.add_argument("--tgt_emb", help="path to the embedding file in the target language")
parser.add_argument("--icp_init_epochs", type=int, default=100)
parser.add_argument("--icp_train_epochs", type=int, default=50)
parser.add_argument("--icp_ft_epochs", type=int, default=50)
parser.add_argument("--n_pca", type=int, default=25)
parser.add_argument("--n_icp_runs", type=int, default=100)
parser.add_argument("--n_init_ex", type=int, default=5000)
parser.add_argument("--n_ft_ex", type=int, default=7500)
parser.add_argument("--n_eval_ex", type=int, default=200000)
parser.add_argument("--n_processes", type=int, default=1)
parser.add_argument("--method", default='csls_knn_10')
parser.add_argument("--cp_dir", default="output")

params = parser.parse_args()

if params.src_emb is not None:
    src_emb = params.src_emb
else:
    src_emb = 'data/wiki.%s.vec' % params.src_lang

if params.tgt_emb is not None:
    tgt_emb = params.tgt_emb
else:
    tgt_emb = 'data/wiki.%s.vec' % params.tgt_lang

src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings(src_emb, params.n_init_ex, False)
np.save('data/%s_%d' % (params.src_lang, params.n_init_ex), src_embeddings)
tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings(tgt_emb, params.n_init_ex, False)
np.save('data/%s_%d' % (params.tgt_lang, params.n_init_ex), tgt_embeddings)

# src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings(src_emb, params.n_ft_ex, False)
# np.save('data/%s_%d' % (params.src_lang, params.n_ft_ex), src_embeddings)
# tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings(tgt_emb, params.n_ft_ex, False)
# np.save('data/%s_%d' % (params.tgt_lang, params.n_ft_ex), tgt_embeddings)

# src_W = np.load("data/%s_%d.npy" % (params.src_lang, params.n_init_ex)).T
# tgt_W = np.load("data/%s_%d.npy" % (params.tgt_lang, params.n_init_ex)).T
src_W = src_embeddings.T
tgt_W = tgt_embeddings.T

data = np.zeros((params.n_icp_runs, 2))

best_idx_x = None
best_idx_y = None

min_rec = 1e8


def run_icp(s0, i):
    np.random.seed(s0 + i)
    icp = ICPTrainer(src_W.copy(), tgt_W.copy(), True, params.n_pca)
    t0 = time.time()
    indices_x, indices_y, rec, bb = icp.train_icp(params.icp_init_epochs)
    dt = time.time() - t0
    print("%d: Rec %f BB %d Time: %f" % (i, rec, bb, dt))
    return indices_x, indices_y, rec, bb


def save_translated(id2word, embeddings, filename):
    f = open(filename, "w")
    f.write(f"{embeddings.shape[0]} {embeddings.shape[1]}\n")
    for i, vec in enumerate(np.around(embeddings, decimals=6)):
        f.write(id2word[i] + ' ' + ' '.join([str(x) for x in vec]) + '\n')
    f.close()


s0 = np.random.randint(50000)
results = []
if params.n_processes == 1:
    for i in range(params.n_icp_runs):
        results += [run_icp(s0, i)]
else:
    pool = multiprocessing.Pool(processes=params.n_processes)
    for result in tqdm.tqdm(pool.imap_unordered(run_icp, range(params.n_icp_runs)), total=params.n_icp_runs):
        results += [result]
    pool.close()


min_rec = 1e8
min_bb = None
for i, result in enumerate(results):
    indices_x, indices_y, rec, bb = result
    data[i, 0] = rec
    data[i, 1] = bb
    if rec < min_rec:
        best_idx_x = indices_x
        best_idx_y = indices_y
        min_rec = rec
        min_bb = bb


idx = np.argmin(data[:, 0], 0)
print("Init - Achieved: Rec %f BB %d" % (data[idx, 0], data[idx, 1]))
icp_train = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])
_, _, rec, bb = icp_train.train_icp(params.icp_train_epochs, True, best_idx_x, best_idx_y)
print("Training - Achieved: Rec %f BB %d" % (rec, bb))
src_W = np.load("data/%s_%d.npy" % (params.src_lang, params.n_ft_ex)).T
tgt_W = np.load("data/%s_%d.npy" % (params.tgt_lang, params.n_ft_ex)).T
icp_ft = ICPTrainer(src_W, tgt_W, False, src_W.shape[0])
icp_ft.icp.TX = icp_train.icp.TX
icp_ft.icp.TY = icp_train.icp.TY
_, _, rec, bb = icp_ft.train_icp(params.icp_ft_epochs, do_reciprocal=True)
print("Reciprocal Pairs - Achieved: Rec %f BB %d" % (rec, bb))
TX = icp_ft.icp.TX
TY = icp_ft.icp.TY

if not os.path.exists(params.cp_dir):
    os.mkdir(params.cp_dir)

# np.save("%s/%s_%s_T" % (params.cp_dir, params.src_lang, params.tgt_lang), TX)
# np.save("%s/%s_%s_T" % (params.cp_dir, params.tgt_lang, params.src_lang), TY)

src_id2word, src_word2id, src_embeddings = utils.read_txt_embeddings(src_emb, params.n_eval_ex, False)
tgt_id2word, tgt_word2id, tgt_embeddings = utils.read_txt_embeddings(tgt_emb, params.n_eval_ex, False)

TranslatedX = src_embeddings.dot(np.transpose(TX))
TranslatedY = tgt_embeddings.dot(np.transpose(TY))

save_translated(src_id2word, TranslatedX, f"{params.cp_dir}/Translated_{params.src_lang}_{params.tgt_lang}.txt")
save_translated(tgt_id2word, TranslatedY, f"{params.cp_dir}/Translated_{params.tgt_lang}_{params.src_lang}.txt")
