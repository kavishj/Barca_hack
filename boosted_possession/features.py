"""Label-free rich observations, half-local context and team renaming."""
import numpy as np
from learned_possession.features import FeatureCache

SIGNED = {'ball_x_home_direction', 'camera_x_home_direction', 'ball_vx_home_direction',
          'distance_margin', 'tag_restart', 'tag_phase'}
CONTEXT = ('ball_visible', 'ball_speed', 'activity', 'distance_margin', 'whistle',
           'home_speed_mean', 'away_speed_mean', 'contact_last_team', 'contact_age_s',
           'reception_last_team', 'reception_age_s', 'blackout_elapsed_s',
           'fine_audio_tonality', 'fine_audio_rise')
SHIFTS = (-10, -3, -1, 1, 3, 10)


def swap_schema(names):
    """An involutive permutation/sign map; no training or reference values."""
    names = tuple(names)
    index = {name:i for i,name in enumerate(names)}
    permutation = np.arange(len(names)); signs = np.ones(len(names), np.float32)
    for i, name in enumerate(names):
        stem, *suffix = name.split('@')
        ending = '@' + suffix[0] if suffix else ''
        if stem.startswith(('home_', 'away_')):
            prefix, tail = stem.split('_', 1)
            other = ('away_' if prefix == 'home' else 'home_') + tail + ending
            if other not in index:raise ValueError('Missing team counterpart: ' + other)
            permutation[i] = index[other]
        elif stem in ('d_home', 'd_away'):
            permutation[i] = index[('d_away' if stem == 'd_home' else 'd_home') + ending]
        elif (stem in SIGNED or stem.startswith('distance_margin_mean_')
              or stem.startswith(('contact_', 'reception_')) and
              any(part in stem for part in ('team', 'balance'))):
            signs[i] = -1
    if not np.array_equal(permutation[permutation], np.arange(len(names))):
        raise ValueError('Team map is not an involution')
    return permutation, signs


def swap(x, names):
    permutation, signs = swap_schema(names)
    return (x[:, permutation] * signs).astype(np.float32)


def build(rich, audio_values, audio_names):
    if audio_values.shape != (len(rich.x), len(audio_names)):
        raise ValueError('Audio timeline and feature names disagree')
    keep = [i for i,n in enumerate(rich.names) if n != 'known_dead_from_metadata']
    columns = [rich.x[:,keep], audio_values]
    names = [rich.names[i] for i in keep] + list(audio_names)
    for seconds in SHIFTS:
        for name in CONTEXT:
            if name not in rich.names:continue
            values = np.full(len(rich.x), np.nan, np.float32)
            offset = round(seconds * rich.fps)
            source = rich.x[:,rich.names.index(name)]
            for _, first, last in rich.halves:
                a, b = max(first, first-offset), min(last+1, last+1-offset)
                if b > a:values[a:b] = source[a+offset:b+offset]
            columns.append(values[:,None]);names.append(name + '@' + str(seconds))
    swap_schema(names)
    return FeatureCache(np.column_stack(columns),tuple(names),rich.fps,rich.halves,
                        dict(rich.provenance,boosted_context=SHIFTS,no_reference_inputs=True))
