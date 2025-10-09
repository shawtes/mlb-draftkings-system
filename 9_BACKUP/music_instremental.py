import pretty_midi
import random

# Create a PrettyMIDI object
song = pretty_midi.PrettyMIDI()

# Set tempo (~150 BPM typical for Lil Uzi Vert)
seconds_per_bar = 2  # ~150 BPM, 4 beats per bar

# Instruments
lead = pretty_midi.Instrument(program=81)   # Synth Lead
bass = pretty_midi.Instrument(program=36)   # Bass
chords = pretty_midi.Instrument(program=88) # Pad for atmospheric chords

# Drums
hi_hat = pretty_midi.Instrument(program=0, is_drum=True)
snare = pretty_midi.Instrument(program=0, is_drum=True)
kick = pretty_midi.Instrument(program=0, is_drum=True)

# Create Melody Loop (C minor pentatonic)
melody_notes = [60, 63, 65, 67, 70]  # C, Eb, F, G, Bb
for bar in range(30):  # ~1 min
    for beat in range(4):
        note_pitch = random.choice(melody_notes)
        start = bar * seconds_per_bar + beat * 0.5
        end = start + 0.4
        lead.notes.append(pretty_midi.Note(velocity=100, pitch=note_pitch, start=start, end=end))

# Atmospheric Chords (hold for 2 bars)
chord_progression = [
    [60, 63, 67],  # Cm
    [65, 68, 72],  # Fm
    [67, 70, 74],  # Gm
    [63, 67, 70],  # Eb
]
for i in range(0, 30, 2):
    chord = chord_progression[(i // 2) % 4]
    for note in chord:
        chords.notes.append(pretty_midi.Note(velocity=60, pitch=note, start=i * seconds_per_bar, end=(i+2) * seconds_per_bar))

# 808 Bass (hits on downbeats)
bass_notes = [36, 38, 40, 36]
for bar in range(30):
    note_pitch = random.choice(bass_notes)
    start = bar * seconds_per_bar
    end = start + 1
    bass.notes.append(pretty_midi.Note(velocity=110, pitch=note_pitch, start=start, end=end))

# Hi-hats (16th notes)
for bar in range(30):
    for i in range(8):
        start = bar * seconds_per_bar + i * 0.25
        hi_hat.notes.append(pretty_midi.Note(velocity=70, pitch=42, start=start, end=start + 0.05))
# Snares (on 2 & 4)
for bar in range(30):
    snare.notes.append(pretty_midi.Note(velocity=100, pitch=38, start=bar * seconds_per_bar + 1.0, end=bar * seconds_per_bar + 1.05))
    snare.notes.append(pretty_midi.Note(velocity=100, pitch=38, start=bar * seconds_per_bar + 3.0, end=bar * seconds_per_bar + 3.05))
# Kicks (on beat 1)
for bar in range(30):
    kick.notes.append(pretty_midi.Note(velocity=120, pitch=36, start=bar * seconds_per_bar, end=bar * seconds_per_bar + 0.1))

# Add all instruments
song.instruments.extend([lead, chords, bass, hi_hat, snare, kick])

# Save to a MIDI file
song.write('lil_uzi_1min_beat.mid')
