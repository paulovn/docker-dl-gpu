"""
Keras native save_weights & load_weights methods choke on empty weights, since
the h5py library can't load/save empty attributes (fixed in master, not yet in
release 2.6). These functions solve the problem by skipping weight management
in layers with no weights (e.g. an Activation or MaxPool layer)
"""

from __future__ import print_function
import sys
import json
import os.path
import numpy as np

from keras import backend as K
import h5py


if sys.version[0] == '3':
    items = lambda x : x.items()
else:
    items = lambda x : x.iteritems()


# --------------------------------------------------------------------------

def save_weights( model, basename ):
    """
    Modification of keras.engine.topology.Container.save_weights to avoid
    saving empty weights
    """
    try:
        f = h5py.File(basename+'.w.h5', 'w')

        if hasattr(model, 'flattened_layers'):
            # support for legacy Sequential/Merge behavior                      
            flattened_layers = model.flattened_layers
        else:
            flattened_layers = model.layers

        f.attrs['layer_names'] = [layer.name.encode('utf8') for layer in flattened_layers]

        for layer in flattened_layers:
            g = f.create_group(layer.name)
            symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
            weight_values = K.batch_get_value(symbolic_weights)
            weight_names = []
            for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
                if hasattr(w, 'name') and w.name:
                    name = str(w.name)
                else:
                    name = 'param_' + str(i)
                weight_names.append(name.encode('utf8'))
            # only add weights attribute if nonempty
            if weight_names:
                g.attrs['weight_names'] = weight_names 
            #else:
            #    g.attrs.if weight_names else np.zeros( (0,), 'S8' ) # ['']
            for name, val in zip(weight_names, weight_values):
                param_dset = g.create_dataset(name, val.shape,
                                              dtype=val.dtype)
                param_dset[:] = val
                #print( weight_names,"=", weight_values)
        f.flush()
    finally:
        f.close()


def load_weights(model, filepath):
    """
    Modification of keras.engine.topology.Container.load_weights to check
    layer weights presence before accessing
    """
    f = h5py.File(filepath, mode='r')

    if hasattr(model, 'flattened_layers'):
        # support for legacy Sequential/Merge behavior
        flattened_layers = model.flattened_layers
    else:
        flattened_layers = model.layers

    if 'nb_layers' in f.attrs:
        # legacy format
        nb_layers = f.attrs['nb_layers']
        if nb_layers != len(flattened_layers):
            raise Exception('You are trying to load a weight file '
                            'containing ' + str(nb_layers) +
                            ' layers into a model with ' +
                            str(len(flattened_layers)) + '.')

        for k in range(nb_layers):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            flattened_layers[k].set_weights(weights)
    else:
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        if len(layer_names) != len(flattened_layers):
            raise Exception('You are trying to load a weight file '
                            'containing ' + str(len(layer_names)) +
                            ' layers into a model with ' +
                            str(len(flattened_layers)) + ' layers.')

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for k, name in enumerate(layer_names):
            g = f[name]
            if 'weight_names' not in g.attrs:
                continue        # skip layer if it has no weights
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]
                layer = flattened_layers[k]
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                if len(weight_values) != len(symbolic_weights):
                    raise Exception('Layer #' + str(k) +
                                    ' (named "' + layer.name +
                                    '" in the current model) was found to '
                                    'correspond to layer ' + name +
                                    ' in the save file. '
                                    'However the new layer ' + layer.name +
                                    ' expects ' + str(len(symbolic_weights)) +
                                    ' weights, but the saved weights have ' +
                                    str(len(weight_values)) +
                                    ' elements.')
                weight_value_tuples += zip(symbolic_weights, weight_values)
        K.batch_set_value(weight_value_tuples)
    f.close()


# --------------------------------------------------------------------------

from keras.callbacks import Callback
from collections import defaultdict
import numpy as np
import time

clock = getattr( time, 'perf_time', time.clock )

class MetricHistory( Callback ):
    '''
    A Keras callback class that tracks all evaluation metrics, 
    both across batches and across epochs
    '''

    def on_train_begin(self, logs={}):
        self.metrics_batch = defaultdict( list )
        self.metrics_epoch = defaultdict( list )
    
    def on_epoch_begin( self, epoch, logs={} ):
        self.epoch_start = clock()
        self.batches = defaultdict( list )

    def on_batch_end(self, batch, logs={}):
        # Store batch metrics
        for k in self.params['metrics']:
            if k in logs:
                self.batches[k].append( logs[k] )

    def on_epoch_end( self, epoch, logs={} ):
        # Store epoch metrics
        for k in self.params['metrics']:
            if k in logs:
                self.metrics_epoch[k].append( logs[k] )
        self.metrics_epoch['time'].append( clock() - self.epoch_start )
        # Consolidate batch metrics
        for k,v in items(self.batches):
            self.metrics_batch[k].append( np.array(v) )
        del self.batches

    def on_train_end(self, logs={}):
        self.metrics_batch = { k : np.array(v) 
                               for k,v in items(self.metrics_batch) }


# --------------------------------------------------------------------------


def history_load( name ):
    f = h5py.File(name+'.h5', 'r')
    h = type( 'SavedHistory', (object,), {} )
    try:
        # Load params
        h.params = {}
        g = f['params']
        for k,v in items(g.attrs):
            h.params[k] = v
        # Load epoch metrics
        g = f['metrics/epoch']
        h.metrics_epoch = { k : np.copy(v) for k,v in items(g) }
        # Load batch metrics
        g = f['metrics/batch']
        h.metrics_batch = { k : np.copy(v) for k,v in items(g) }
    finally:
        f.close()
    return h


def history_save( history, name ):
    f = h5py.File(name+'.h5', 'w')
    try:
        # Load params
        g = f.create_group('params')
        for k,v in items(history.params):
            g.attrs[k] = v
        # Save epoch metrics
        g = f.create_group('metrics/epoch')
        for n,m in items(history.metrics_epoch):
            dat = g.create_dataset(n, data=np.array(m) )
        # Save batch metrics
        g = f.create_group('metrics/batch')
        for n,m in items(history.metrics_batch):
            dat = g.create_dataset(n, data=np.array(m) )
        f.flush()
    finally:
        f.close()


# --------------------------------------------------------------------------

def model_save( model, basename, history=None ):
    """
    Save a full model: architecture and weights, into a file
      @param model (Model): the Keras model to save
      @param basename (str): filename to use. Two files will be written: a
        JSON file (model architecture) and an HDF5 file (model weights)
      @param history (History): optional training history to save
    """
    with open( basename+'.m.json', 'w') as f:
        f.write( model.to_json() )
    save_weights( model, basename )
    if history:
        history_save( history, basename + '.h' )


def model_load( basename, compile={}, history=True ):
    """
    Load a model saved with model_save(): structure & weights will be
    restored
      @param basename (str): basename used to save it
      @param compile (dict): arguments to be used when compiling the model
      @param history (bool): load also training history, if available
    """
    from keras.models import model_from_json
    model = model_from_json(open(basename+'.m.json').read())
    model.compile( **compile )
    load_weights( model, basename+'.w.h5' )
    if not history:
        return model
    elif not os.path.exists( basename + '.h.h5' ):
        return model, None
    return model, history_load( basename + '.h' )


