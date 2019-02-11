import numpy
import pickle
import glob

from music21 import corpus, converter

from keras.models import Model
from keras.layers import LSTM, Input, Dropout, Dense, Activation, Embedding, Concatenate
from keras.optimizers import RMSprop
from keras.utils import np_utils

def get_music_list(name):
    
    if name == 'bach':
        file_list = ['bwv' + str(x['bwv']) for x in corpus.chorales.ChoraleList().byBWV.values()]
        parser = corpus
    elif name == 'local':
        file_list = glob.glob("midi_songs/*.mid")
        parser = converter
    
    return file_list, parser

def create_network(n_notes, n_durations, seq_len = None, embed_size = 100):
    """ create the structure of the neural network """

    notes_in = Input(shape = (seq_len,))
    durations_in = Input(shape = (seq_len,))

    x1 = Embedding(n_notes, embed_size)(notes_in)
    x2 = Embedding(n_durations, embed_size)(durations_in) 

    x = Concatenate()([x1,x2])

    x = LSTM(256, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(256)(x)
    x = Dropout(0.2)(x)
    notes_out = Dense(n_notes, activation = 'softmax')(x)
    durations_out = Dense(n_durations, activation = 'softmax')(x)

    model = Model([notes_in, durations_in], [notes_out, durations_out])

    # model.summary()

    opti = RMSprop(lr = 0.001)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=opti)

    return model


def get_distinct(elements):
    # Get all pitch names
    element_names = sorted(set(elements))
    n_elements = len(element_names)
    return (element_names, n_elements)

def create_lookups(element_names):
    # create dictionary to map notes and durations to integers
    element_to_int = dict((element, number) for number, element in enumerate(element_names))
    int_to_element = dict((number, element) for number, element in enumerate(element_names))

    return (element_to_int, int_to_element)
    

def prepare_sequences(notes, durations, lookups, distincts):
    """ Prepare the sequences used to train the Neural Network """
    sequence_length = 32

    note_to_int, int_to_note, duration_to_int, int_to_duration = lookups
    note_names, n_notes, duration_names, n_durations = distincts

    notes_network_input = []
    notes_network_output = []
    durations_network_input = []
    durations_network_output = []

    # create input sequences and the corresponding outputs
    for i in range(len(notes) - sequence_length):
        notes_sequence_in = notes[i:i + sequence_length]
        notes_sequence_out = notes[i + sequence_length]
        notes_network_input.append([note_to_int[char] for char in notes_sequence_in])
        notes_network_output.append(note_to_int[notes_sequence_out])

        durations_sequence_in = durations[i:i + sequence_length]
        durations_sequence_out = durations[i + sequence_length]
        durations_network_input.append([duration_to_int[char] for char in durations_sequence_in])
        durations_network_output.append(duration_to_int[durations_sequence_out])

    n_patterns = len(notes_network_input)

    # reshape the input into a format compatible with LSTM layers
    notes_network_input = numpy.reshape(notes_network_input, (n_patterns, sequence_length))
    durations_network_input = numpy.reshape(durations_network_input, (n_patterns, sequence_length))
    network_input = [notes_network_input, durations_network_input]

    notes_network_output = np_utils.to_categorical(notes_network_output, num_classes=n_notes)
    durations_network_output = np_utils.to_categorical(durations_network_output, num_classes=n_durations)
    network_output = [notes_network_output, durations_network_output]

    return (network_input, network_output)
