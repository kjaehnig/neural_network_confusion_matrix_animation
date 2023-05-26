import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
# from utils import img_gen
import seaborn as sns
import imageio
from tqdm import tqdm
from matplotlib import cm
import matplotlib as mpl
from matplotlib.pyplot import rcParams


rcParams['font.size'] = 16
rcParams['font.weight'] = 'bold'

def DNNmodel():

    mdl = Sequential()

    mdl.add(Flatten(input_shape=(28,28)))

    mdl.add(Dense(
        256,
        activation='relu',
        kernel_initializer='normal'))

    mdl.add(Dense(
        512,
        activation='relu',
        # kernel_initializer='normal'
        ))

    mdl.add(Dropout(0.5))

    mdl.add(Dense(1024, activation='relu',
        # kernel_regularizer='l2'
        ))

    mdl.add(Dropout(0.50))

    # mdl.add(Dense(1024, activation='relu',
    #     kernel_regularizer='l2'))

    mdl.add(
        Dense(units=10, activation='softmax')
        )
 
    return mdl


def main():
    mnist = np.load("/Users/karljaehnig/Repositories/neural_network_confusion_matrix_animation/mnist_28x28_data.npz")

    train_X = mnist['train_X']
    train_y = mnist['train_y']
    test_X = mnist['test_X']
    test_y = mnist['test_y']

    mdl = DNNmodel()

    # mdl.summary()

    mdl.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy'],
        experimental_run_tf_function=False
        )

    frames = 180
    images = []

    nclasses = np.unique(test_y).shape[0]
    spe = 2
    minacc = 0.0
    print('starting NN run')
    for ii in tqdm(range(frames)):
        mdl.fit(train_X,train_y,
                steps_per_epoch=spe,
                epochs=1,
                verbose=0,
                batch_size=4)
        ii_acc = mdl.evaluate(test_X, test_y, verbose=0)[1]

        if ii_acc > minacc:
            minacc = ii_acc

        cmat = confusion_matrix(test_y,np.argmax(mdl.predict(test_X,verbose=0),axis=1))



        perfectcm = np.diag(np.ones(nclasses))
        pcm_annot = perfectcm.astype("str")
        pcm_annot[pcm_annot=='0.0'] = ''


        fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize=(24,10), gridspec_kw={'width_ratios': [4,4,0.25]})


        cb = mpl.colorbar.ColorbarBase(ax3, orientation='vertical',
                               cmap=cm.magma,
                               norm=mpl.colors.Normalize(0, 1),  # vmax and vmin
                               extend='neither',
                               label='Fraction',
                               ticks=[0,.25,.5,.75, 1])

        cmatnorm = np.around(cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis], decimals=2)
        cmat_diag = cmatnorm[np.diag_indices_from(cmatnorm)]
        cmat_annot = np.diag(np.round(cmat_diag,2).astype('str'))
        cmat_annot[cmat_annot=='0.0'] = ''

        sns.heatmap(cmatnorm,
            annot=cmat_annot,
            xticklabels=np.arange(10),
            yticklabels=np.arange(10),
            ax=ax2,
            cbar=False,
            annot_kws={'fontweight':'bold',
                        'fontsize':16},
            vmin=0,
            vmax=1.0,
            square=False,
            rasterized=True,
            fmt='')

        sns.heatmap(perfectcm,
            annot=pcm_annot,
            xticklabels=np.arange(10),
            yticklabels=np.arange(10),
            ax=ax1,
            cbar=False,
            annot_kws={'fontweight':'bold',
                        'fontsize':16},
            vmin=0,
            vmax=1.0,
            square=False,
            rasterized=True,
            fmt='')

        ax1.set_title("A 'perfect' classifier model", fontweight='bold')

        ax1.set_xlabel("Predicted Digit", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Actual Digit", fontsize=16, fontweight='bold')
        ax2.set_xlabel("Predicted Digit", fontsize=16, fontweight='bold')
        ax2.set_ylabel("Actual Digit", fontsize=16, fontweight='bold')

        ax2.set_title(f"Dense NN (MNIST), epoch {ii+1}, accuracy {np.round(minacc,2)}", fontweight='bold')
        fn = f"/Users/karljaehnig/Repositories/neural_network_confusion_matrix_animation/confusion_matrix_images/cm{str(ii).zfill(4)}.png"
        plt.savefig(fn, bbox_inches='tight', dpi=75)
        plt.close()
        images.append(imageio.imread(fn))


    print('generating CM evolution gif')
    imageio.mimsave('confusion_matrix_evolution.gif', images, fps=10)





if __name__ == "__main__":
    main()


    # history_mdl = mdl.fit(
    #     trgen,
    #     epochs=100,
    #     verbose=1
    #     )