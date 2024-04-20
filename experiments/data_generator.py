import numpy as np
import os
import itertools


# A: 48 classes (excluding special characters)
# notes1 = map(''.join, itertools.product(['c', 'd'], ['1', '2', '4', '8']))
# notes2 = map(''.join, itertools.product(['e', 'f', 'g', 'a', 'b'], ['', ','], ['1', '2', '4', '8']))
# ALL_NOTES = list(notes1) + list(notes2)

# B: 84 classes (excluding special characters)
# octaves = ['', ',']
# accidentals = ['', 'is']
# notes1 = list(map(''.join, itertools.product(['e', 'b'], octaves)))
# notes2 = list(map(''.join, itertools.product(['c', 'd'], accidentals)))
# notes3 = list(map(''.join, itertools.product(['f', 'g', 'a'], accidentals, octaves)))
# subdivisions = ['1', '2', '4', '8']
# ALL_NOTES = list(map(''.join, itertools.product(notes1 + notes2 + notes3 + ['r'], subdivisions)))

# C: 105 classes
octaves = ['', ',']
accidentals = ['', 'is']
notes1 = list(map(''.join, itertools.product(['e', 'b'], octaves)))
notes2 = list(map(''.join, itertools.product(['c', 'd'], accidentals)))
notes3 = list(map(''.join, itertools.product(['f', 'g', 'a'], accidentals, octaves)))
subdivisions = ['1', '2', '4', '8', '16']
ALL_NOTES = list(map(''.join, itertools.product(notes1 + notes2 + notes3 + ['r'], subdivisions)))


# Generate random note
def random_note():
    return np.random.choice(ALL_NOTES)


def generate_score(m=20):
    start = '{'
    end = '}'
    score = [start]
    for i in range(m):
        note = random_note()
        # Don't allow leading, training, and consecutive rest
        while note[0] == 'r' and (i in [0, m-1] or score[-1][0] == 'r'):
            note = random_note()
        score.append(note)
    score.append(end)
    score_str = ' '.join(score)
    return score_str


def to_full_score(notes_str, name='', version='2.24.3'):
    # Random bpm for generation. Fixed at 120 for now
    bpm = 120

    ly = f'''\\version "{version}"
    \\header {{
        title = "{name}"
        composer = "Auto Bass Transcriber"
    }}
    \\score {{
        \\layout {{
            \\tempo 4 = {bpm}
            \\clef bass
        }}

        \\new Staff {{
            \\absolute {notes_str}
        }}

        \\midi {{ }}
    }}
    '''

    return ly


if __name__ == '__main__':
    # Directories
    base_dir = os.path.join(os.path.dirname(__file__), 'audioC')
    labels_dir = os.path.join(base_dir, 'labels')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # Generate B batches of N scores
    B = 40
    N = 250
    np.random.seed(0)
    for batch_num in range(B):
        # New ly and midi directories for each batch
        ly_dir = os.path.join(base_dir, f'ly_{batch_num}')
        midi_dir = os.path.join(base_dir, f'midi_{batch_num}')
        if not os.path.exists(ly_dir):
            os.makedirs(ly_dir)
        if not os.path.exists(midi_dir):
            os.makedirs(midi_dir)

        M = np.random.randint(10, 20)
        scores = [generate_score(M) for _ in range(N)]
        for i, score in enumerate(scores):
            filename = f'{i + batch_num * N}'
            # Write string as label to file
            print(f'Writing label for {os.path.join(labels_dir, f"{filename}.txt")}')
            with open(os.path.join(labels_dir, f'{filename}.txt'), 'w') as f:
                f.write(score)

            # Generate full .ly score for MIDI
            full_score = to_full_score(score, name=filename)
            print(f'Writing full score for {os.path.join(ly_dir, f"{filename}.ly")}')
            with open(os.path.join(ly_dir, f'{filename}.ly'), 'w') as f:
                f.write(full_score)
