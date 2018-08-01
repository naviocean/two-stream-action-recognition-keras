"""
Train our temporal-stream CNN on optical flow frames.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from spatial_train_model import get_model, freeze_all_but_top, freeze_all_but_mid_and_top
from spatial_train_data import DataSet, get_generators
import time
import os.path
from os import makedirs
from keras.utils import multi_gpu_model
from multi_gpu_checkpoint import MultiGPUCheckpointCallback

import warnings


def train_model(parallel_model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    parallel_model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        validation_data=validation_generator,
        validation_steps=10,
        epochs=nb_epoch,
        callbacks=callbacks)
    return parallel_model

def train(num_of_snip=5, saved_weights=None,
        class_limit=None, image_shape=(224, 224),
        load_to_memory=False, batch_size=32, nb_epoch=100, name_str=None):

    # Get local time.
    time_str = time.strftime("%y%m%d%H%M", time.localtime())

    if name_str == None:
        name_str = time_str

    print("class_limit = ", class_limit)

    if image_shape is None:
        data = DataSet(
                class_limit=class_limit
                )
    else:
        data = DataSet(
                image_shape=image_shape,
                class_limit=class_limit
                )
    
    # Get generators.
    generators = get_generators(data=data, image_shape=image_shape, batch_size=batch_size)

    # Get the model.
    model = get_model(data=data)

    # Callbacks: Save the model.
    directory1 = os.path.join('out', 'checkpoints', name_str)
    if not os.path.exists(directory1):
        os.makedirs(directory1)
    checkpointer = MultiGPUCheckpointCallback(
        filepath=os.path.join(directory1, 'spatial-{epoch:03d}-{val_loss:.3f}.hdf5'),
        base_model = model,
        verbose=1,
        save_best_only=True)

    # Callbacks: TensorBoard
    directory2 = os.path.join('out', 'TB', name_str)
    if not os.path.exists(directory2):
        os.makedirs(directory2)
    tb = TensorBoard(log_dir=os.path.join(directory2))

    # Callbacks: Early stoper
    early_stopper = EarlyStopping(monitor='loss', patience=100)

    # Callbacks: Save results.
    directory3 = os.path.join('out', 'logs', name_str)
    if not os.path.exists(directory3):
        os.makedirs(directory3)
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(directory3, 'training-' + \
        str(timestamp) + '.log'))


    parallel_model = multi_gpu_model(model, gpus=2)
    if saved_weights is None:
        print("Loading network from ImageNet weights.")
        print("Get and train the top layers...")
        parallel_model, model = freeze_all_but_top(parallel_model, model)
        parallel_model = train_model(parallel_model, 2, generators)
    else:
        print("Loading saved model: %s." % saved_weights)
        model.load_weights(saved_weights)

    print("Get and train the mid layers...")
    parallel_model, model = freeze_all_but_mid_and_top(parallel_model, model)
    parallel_model = train_model(parallel_model, 2, generators, [tb, early_stopper, csv_logger, checkpointer])

def main():
    """These are the main training settings. Set each before running
    this file."""
    "=============================================================================="
    saved_weights = None#'/home/duong/two-stream-action-recognition-keras/out/checkpoints/1807301247/010-2.174.hdf5'
    class_limit = None  # int, can be 1-101 or None
    num_of_snip = 1 # number of chunks used for each video
    image_shape=(224, 224)
    load_to_memory = True  # pre-load the sequencea in,o memory
    batch_size = 512
    nb_epoch = 500
    name_str = 'test'
    "=============================================================================="
    warnings.filterwarnings("always")
    train(num_of_snip=num_of_snip, saved_weights=saved_weights,
            class_limit=class_limit, image_shape=image_shape,
            load_to_memory=load_to_memory, batch_size=batch_size,
            nb_epoch=nb_epoch, name_str=name_str)

if __name__ == '__main__':
    main()
