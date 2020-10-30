import numpy as np
import pandas as pd
import os
import networkx as nx
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

from spektral.layers import GraphConv

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import tensorflow as tf
from tensorflow.keras.regularizers import l2

from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json

def read_cora_data(fp):
    data, edges = [], []
    for ro, di, files in os.walk(fp):
        for file in files:
            if '.content' in file:
                with open(os.path.join(ro, file),'r') as f:
                    data.extend(f.read().splitlines())
            elif 'cites' in file:
                with open(os.path.join(ro, file),'r') as f:
                    edges.extend(f.read().splitlines())
    data = shuffle(data)
    return data, edges

def read_twitch_data(fp):
    data, edges = [], []

    for ro, di, files in os.walk(fp):
        for file in files:
            if 'edges' in file:
                with open(os.path.join(ro, file), 'r') as f:
                    edges.extend(f.read().splitlines())
            elif 'features.json' in file:
                with open(os.path.join(ro, file),'r') as json_data:
                    j_data = json.load(json_data)
                    d = pd.DataFrame.from_dict(j_data, orient='index')
                    d['index'] = d.index
                    d['index'] = d['index'].astype(int)
                    d = d.reset_index(drop=True)
                    d = d.replace(np.NaN, 0)
            elif 'target.csv' in file:
                target_df = pd.read_csv(os.path.join(ro, file))
                target_df = target_df[['new_id', 'partner']]

    edges = edges[1:]
    combined = pd.merge(d, target_df, left_on='index', right_on='new_id').drop(['new_id'], axis=1)
    order = [combined.columns[-2]] + list(combined.columns[0:-2]) + [combined.columns[-1]]
    combined = combined[order]

    combined.to_csv(os.path.join(fp, 'features.csv'), '\t')

    for ro, di, files in os.walk(fp):
        for file in files:
            if 'features.csv' in file:
                with open(os.path.join(ro, file), 'r') as f:
                    data.extend(f.read().splitlines())

    data = data[1:]

    return data, edges


def read_facebook_data(fp):
    data, edges = [], []

    for ro, di, files in os.walk(fp):
        for file in files:
            if 'edges' in file:
                with open(os.path.join(ro, file), 'r') as f:
                    edges.extend(f.read().splitlines())
            elif 'features.json' in file:
                with open(os.path.join(ro, file),'r') as json_data:
                    j_data = json.load(json_data)
                    d = pd.DataFrame.from_dict(j_data, orient='index')
                    d['index'] = d.index
                    d['index'] = d['index'].astype(int)
                    d = d.reset_index(drop=True)
                    d = d.replace(np.NaN, 0)
            elif 'target.csv' in file:
                target_df = pd.read_csv(os.path.join(ro, file))
                target_df = target_df[['id', 'page_type']]

    edges = edges[1:]
    combined = pd.merge(d, target_df, left_on='index', right_on='id').drop(['id'], axis=1)
    order = [combined.columns[-2]] + list(combined.columns[0:-2]) + [combined.columns[-1]]
    combined = combined[order]

    combined.to_csv(os.path.join(fp, 'features.csv'), '\t')

    for ro, di, files in os.walk(fp):
        for file in files:
            if 'features.csv' in file:
                with open(os.path.join(ro, file), 'r') as f:
                    data.extend(f.read().splitlines())

    data = data[1:]

    return data, edges


def parse_data(data):
    labels, nodes, X = [], [], []
    for i, data in enumerate(data):
        features = data.split('\t')

        labels.append(features[-1])
        X.append(features[1:-1])
        nodes.append(features[0])

    X = np.array(X, dtype=float)
    X = np.array(X, dtype=int)
    return labels, nodes, X, X.shape[0], X.shape[1]

def parse_edges(edges):
    edge_list = []
    for edge in edges:
        e = edge.split('\t')
        edge_list.append((e[0],e[1]))
    return edge_list

def limit_data(labels,limit=20,val_num=500,test_num=1000):
    '''
    Get the index of train, validation, and test data
    '''
    label_counter = dict((l, 0) for l in labels)
    train_idx = []

    for i in range(len(labels)):
        label = labels[i]
        if label_counter[label]<limit:
            #add the example to the training data
            train_idx.append(i)
            label_counter[label]+=1

        #exit the loop once we found 20 examples for each class
        if all(count == limit for count in label_counter.values()):
            break

    #get the indices that do not go to traning data
    rest_idx = [x for x in range(len(labels)) if x not in train_idx]
    #get the first val_num
    val_idx = rest_idx[:val_num]
    test_idx = rest_idx[val_num:(val_num+test_num)]
    return train_idx, val_idx,test_idx

def encode_label(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return labels, label_encoder.classes_

def build_adj(nodes, edge_list):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)

    #obtain the adjacency matrix (A)
    A = nx.adjacency_matrix(G)
    return A

def GCN(A, F, N, X, train_mask, val_mask, labels_encoded, num_classes, channels=16, dropout=0.5, l2_reg=5e-4, learning_rate=1e-2, epochs=200, es_patience=10):
    A = GraphConv.preprocess(A).astype('f4')

    X_in = Input(shape=(F, ))
    fltr_in = Input((N, ), sparse=True)

    dropout_1 = Dropout(dropout)(X_in)
    graph_conv_1 = GraphConv(channels,
                         activation='relu',
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False)([dropout_1, fltr_in])

    dropout_2 = Dropout(dropout)(graph_conv_1)
    graph_conv_2 = GraphConv(num_classes,
                         activation='softmax',
                         use_bias=False)([dropout_2, fltr_in])

    model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  weighted_metrics=['acc'])

    tbCallBack_GCN = tf.keras.callbacks.TensorBoard(
        log_dir='./Tensorboard_GCN_cora',
    )
    callback_GCN = [tbCallBack_GCN]

    validation_data = ([X, A], labels_encoded, val_mask)
    model.fit([X, A],
          labels_encoded,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,
          callbacks=[
              EarlyStopping(patience=es_patience,  restore_best_weights=True),
              tbCallBack_GCN
          ])

def complete(fp, source):
    if source == "cora":
        data, edges = read_cora_data(fp)
    elif source == "facebook":
        data, edges = read_facebook_data(fp)
    elif source == "twitch":
        data, edges = read_twitch_data(fp)

    labels, nodes, X, N, F = parse_data(data)
    edge_list = parse_edges(edges)
    train_idx,val_idx,test_idx = limit_data(labels)

    train_mask = np.zeros((N,),dtype=bool)
    train_mask[train_idx] = True

    val_mask = np.zeros((N,),dtype=bool)
    val_mask[val_idx] = True

    test_mask = np.zeros((N,),dtype=bool)
    test_mask[test_idx] = True

    labels_encoded, classes = encode_label(labels)

    A = build_adj(nodes, edge_list)

    GCN(A, F, N, X, train_mask, val_mask, labels_encoded, len(set(labels)))
